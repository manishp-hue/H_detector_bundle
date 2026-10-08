#include "tracker.hpp"

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/rtsp/gstrtspconnection.h>
#include <gio/gio.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>

namespace {
using namespace std::chrono_literals;
using helipad::CandidateSource;
using helipad::HelipadTracker;
using helipad::TrackResult;
using helipad::TrackState;
using helipad::VisionConfig;

volatile std::sig_atomic_t g_stop = 0;
void onSignal(int) { g_stop = 1; }

std::uint64_t monoMs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Config {
    int sensorId = 0;
    int width = 1920;
    int height = 1080;
    int fps = 60;
    int streamWidth = 1280;
    int streamHeight = 720;
    int streamFps = 30;
    int bitrateKbps = 3000;
    int controlPort = 5701;
    int rtspPort = 8554;
    std::string localDisplay = "auto";
    std::string dinoPlan;
    std::string samPlan;
};

void usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " [options]\n"
        << "  --sensor-id 0\n"
        << "  --width 1920 --height 1080 --fps 60\n"
        << "  --stream-width 1280 --stream-height 720 --stream-fps 30\n"
        << "  --bitrate 3000\n"
        << "  --control-port 5701 --rtsp-port 8554\n"
        << "  --local-display auto|on|off\n"
        << "  --dino-plan path --sam-plan path\n";
}

bool parseArgs(int argc, char** argv, Config& c) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (k == "--sensor-id") c.sensorId = std::stoi(need("--sensor-id"));
        else if (k == "--width") c.width = std::stoi(need("--width"));
        else if (k == "--height") c.height = std::stoi(need("--height"));
        else if (k == "--fps") c.fps = std::stoi(need("--fps"));
        else if (k == "--stream-width") c.streamWidth = std::stoi(need("--stream-width"));
        else if (k == "--stream-height") c.streamHeight = std::stoi(need("--stream-height"));
        else if (k == "--stream-fps") c.streamFps = std::stoi(need("--stream-fps"));
        else if (k == "--bitrate") c.bitrateKbps = std::stoi(need("--bitrate"));
        else if (k == "--control-port") c.controlPort = std::stoi(need("--control-port"));
        else if (k == "--rtsp-port") c.rtspPort = std::stoi(need("--rtsp-port"));
        else if (k == "--local-display") c.localDisplay = need("--local-display");
        else if (k == "--dino-plan") c.dinoPlan = need("--dino-plan");
        else if (k == "--sam-plan") c.samPlan = need("--sam-plan");
        else if (k == "-h" || k == "--help") { usage(argv[0]); return false; }
        else throw std::runtime_error("unknown option: " + k);
    }
    c.streamFps = std::clamp(c.streamFps, 1, 30);
    c.bitrateKbps = std::clamp(c.bitrateKbps, 500, 12000);
    if (c.localDisplay != "auto" && c.localDisplay != "on" && c.localDisplay != "off")
        throw std::runtime_error("--local-display must be auto, on, or off");
    return true;
}

std::string mipiPipeline(const Config& c) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(c.sensorId) +
        " ! video/x-raw(memory:NVMM),width=" + std::to_string(c.width) +
        ",height=" + std::to_string(c.height) +
        ",framerate=" + std::to_string(c.fps) + "/1,format=NV12 "
        "! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
        "! nvvidconv ! video/x-raw,format=BGRx "
        "! videoconvert n-threads=2 ! video/x-raw,format=BGR "
        "! appsink drop=true max-buffers=1 sync=false";
}

const char* stateName(TrackState s) {
    switch (s) {
        case TrackState::Searching: return "SEARCHING";
        case TrackState::Confirming: return "CONFIRMING";
        case TrackState::Locked: return "LOCKED";
        case TrackState::Suspect: return "SUSPECT";
    }
    return "UNKNOWN";
}

cv::Scalar stateColor(TrackState s) {
    switch (s) {
        case TrackState::Locked: return {0,255,0};
        case TrackState::Confirming: return {0,255,255};
        case TrackState::Searching: return {0,165,255};
        case TrackState::Suspect: return {0,140,255};
    }
    return {255,255,255};
}

cv::Rect2f clampRect(const cv::Rect2f& r, int w, int h) {
    const float x1 = std::clamp(r.x, 0.0f, static_cast<float>(w));
    const float y1 = std::clamp(r.y, 0.0f, static_cast<float>(h));
    const float x2 = std::clamp(r.x + r.width, 0.0f, static_cast<float>(w));
    const float y2 = std::clamp(r.y + r.height, 0.0f, static_cast<float>(h));
    if (x2 <= x1 || y2 <= y1) return {};
    return {x1,y1,x2-x1,y2-y1};
}

float uiScaleFor(const cv::Mat& frame) {
    return std::clamp(frame.rows / 1080.0f, 0.55f, 1.65f);
}

void putOutlinedText(cv::Mat& image, const std::string& text, cv::Point org,
                     double scale, cv::Scalar color, int thickness = 2) {
    cv::putText(image, text, org, cv::FONT_HERSHEY_SIMPLEX, scale,
                {0,0,0}, thickness + 3, cv::LINE_AA);
    cv::putText(image, text, org, cv::FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv::LINE_AA);
}

double processedPercent(double procFps, double sourceFps) {
    if (sourceFps <= 1e-6) return 0.0;
    return 100.0 * procFps / sourceFps;
}

void annotateMain(cv::Mat& frame, const TrackResult& r,
                  double procFps, double sourceFps, double trackFps,
                  int frameNo) {
    (void)trackFps;
    const float us = uiScaleFor(frame);
    const cv::Point imageCenter(frame.cols/2, frame.rows/2);
    cv::drawMarker(frame, imageCenter, {255,255,0}, cv::MARKER_CROSS,
                   int(24*us), std::max(2,int(2*us)));
    const bool showTarget = r.hasTarget || r.hasPrediction;
    if (showTarget) {
        const bool locked = r.guidanceValid;
        const bool suspect = r.state == TrackState::Suspect;
        const cv::Scalar col = locked ? cv::Scalar(0,255,0)
            : (suspect ? cv::Scalar(0,140,255) : cv::Scalar(0,255,255));
        cv::Rect box = cv::Rect(clampRect(r.target.box, frame.cols, frame.rows));
        cv::rectangle(frame, box, col, std::max(2,int(2*us)));
        cv::drawMarker(frame, r.target.center, col, cv::MARKER_CROSS,
                       int(28*us), std::max(2,int(3*us)));
        if (locked) cv::line(frame, imageCenter, r.target.center, {255,0,255},
                             std::max(2,int(2*us)));
        const std::string tag = std::string(locked ? "LOCKED " : (suspect ? "SUSPECT " : "CANDIDATE "))
            + helipad::sourceName(r.target.source);
        putOutlinedText(frame, tag,
                        {box.x, std::max(int(58*us), box.y-int(10*us))},
                        0.62*us, col, std::max(1,int(2*us)));
    }
    std::ostringstream hud;
    hud << stateName(r.state)
        << "  FPS=" << std::fixed << std::setprecision(1) << procFps << "/" << sourceFps
        << "  PROC %=" << std::setprecision(0) << processedPercent(procFps,sourceFps)
        << "  AI=" << (r.gdinoOn && r.sam2On ? "ON" : "OFF");
    if (r.guidanceValid) {
        hud << "  H=" << std::setprecision(2) << r.target.score
            << "  EX=" << std::showpos << std::setprecision(0) << r.guidance.errorX << "px"
            << "  EY=" << r.guidance.errorY << "px"
            << "  R=" << std::setprecision(2) << r.guidance.roll
            << "  P=" << r.guidance.pitch << std::noshowpos;
    }
    if (r.identityAgeMs >= 0.0f)
        hud << "  IDAGE=" << std::setprecision(0) << r.identityAgeMs << "ms";
    putOutlinedText(frame, hud.str(), {int(18*us),int(34*us)}, 0.62*us,
                    {0,255,255}, std::max(1,int(2*us)));
    putOutlinedText(frame, "FRAME " + std::to_string(frameNo),
                    {frame.cols-int(190*us),int(34*us)}, 0.46*us,
                    {255,255,255}, std::max(1,int(1*us)));
}

int automaticPanelWidth(const cv::Mat& frame) {
    const bool low = frame.cols <= 800 || frame.rows <= 480;
    const int minW = low ? 210 : 280;
    const int maxW = std::max(minW, std::min(720, frame.cols-20));
    return std::clamp(int(std::round(frame.cols*0.24)), minW, maxW);
}

void drawTrackingInset(cv::Mat& output, const cv::Mat& original,
                       const TrackResult& r, double procFps, double sourceFps,
                       double trackFps, int requestedWidth) {
    const bool showTarget = r.hasTarget || r.hasPrediction;
    if (output.empty() || original.empty()) return;
    const int h=original.rows,w=original.cols;
    const bool low=w<=800 || h<=480;
    const float us=std::clamp(h/1080.0f, low?0.55f:0.70f, 1.65f);
    const int margin=std::max(low?6:10,int(14*us));
    int iw=requestedWidth>0?requestedWidth:automaticPanelWidth(original);
    const int minIw=low?210:280;
    iw=std::clamp(iw,minIw,std::max(minIw,w-2*margin));
    const int titleH=std::max(low?22:30,int(34*us));
    const int zoomH=std::max(low?100:150,int(std::round(iw*0.56)));
    const int lineH=std::max(low?14:20,int(23*us));
    const int infoH=lineH*11+int(18*us);
    int ih=std::min(titleH+zoomH+infoH,h-2*margin);
    const int x0=w-iw-margin;
    const int y0=std::max(margin,int(42*us));
    if(x0<0 || y0+ih>h) return;
    const cv::Rect outer(x0,y0,iw,ih);
    cv::Mat overlay=output.clone();
    cv::rectangle(overlay,outer,{0,0,0},cv::FILLED);
    cv::addWeighted(overlay,0.72,output,0.28,0.0,output);
    cv::rectangle(output,outer,{210,210,210},std::max(1,int(1*us)));
    putOutlinedText(output,"TRACKING VIEW",{x0+margin,y0+int(25*us)},
                    0.55*us,{255,255,255},std::max(1,int(1*us)));

    cv::Point2f tc(w*0.5f,h*0.5f);
    float span=0.0f;
    if(showTarget){tc=r.target.center;span=std::max({r.target.span,r.target.box.width,r.target.box.height});}
    int cropSize=showTarget?int(std::round(std::max(110.0f*us,span*2.8f)))
                           :int(std::round(std::min(w,h)*0.42f));
    cropSize=std::clamp(cropSize,100,std::max(w,h));
    int sx1=int(std::round(tc.x))-cropSize/2, sy1=int(std::round(tc.y))-cropSize/2;
    int sx2=sx1+cropSize, sy2=sy1+cropSize;
    if(sx1<0){sx2-=sx1;sx1=0;} if(sy1<0){sy2-=sy1;sy1=0;}
    if(sx2>w){const int d=sx2-w;sx1-=d;sx2=w;} if(sy2>h){const int d=sy2-h;sy1-=d;sy2=h;}
    sx1=std::clamp(sx1,0,w-1); sy1=std::clamp(sy1,0,h-1);
    sx2=std::clamp(sx2,sx1+1,w); sy2=std::clamp(sy2,sy1+1,h);
    const cv::Rect cropRect(sx1,sy1,sx2-sx1,sy2-sy1);
    cv::Mat crop=original(cropRect);
    const int zx=x0+margin, zy=y0+titleH, zw=std::max(1,iw-2*margin), zh=std::max(1,zoomH-margin/2);
    const float fit=std::min(zw/float(crop.cols),zh/float(crop.rows));
    const int nw=std::max(1,int(std::round(crop.cols*fit))), nh=std::max(1,int(std::round(crop.rows*fit)));
    const int px=zx+(zw-nw)/2, py=zy+(zh-nh)/2;
    cv::Mat zoom; cv::resize(crop,zoom,{nw,nh},0,0,fit>=1.0f?cv::INTER_LANCZOS4:cv::INTER_AREA);
    zoom.copyTo(output(cv::Rect(px,py,nw,nh))); cv::rectangle(output,{px,py,nw,nh},{180,180,180},1);
    if(showTarget){
        const cv::Rect2f b=clampRect(r.target.box,w,h);
        int rx1=int(std::round((b.x-sx1)*fit))+px, ry1=int(std::round((b.y-sy1)*fit))+py;
        int rx2=int(std::round((b.x+b.width-sx1)*fit))+px, ry2=int(std::round((b.y+b.height-sy1)*fit))+py;
        rx1=std::clamp(rx1,px,px+nw-1); ry1=std::clamp(ry1,py,py+nh-1);
        rx2=std::clamp(rx2,rx1+1,px+nw); ry2=std::clamp(ry2,ry1+1,py+nh);
        const cv::Scalar col=r.guidanceValid?cv::Scalar(0,255,0):(r.state==TrackState::Suspect?cv::Scalar(0,140,255):cv::Scalar(0,255,255));
        cv::rectangle(output,{rx1,ry1},{rx2,ry2},col,std::max(1,int(2*us)));
        const int mcx=int(std::round((tc.x-sx1)*fit))+px, mcy=int(std::round((tc.y-sy1)*fit))+py;
        if(mcx>=px&&mcx<px+nw&&mcy>=py&&mcy<py+nh)
            cv::drawMarker(output,{mcx,mcy},{0,0,255},cv::MARKER_CROSS,int(16*us),std::max(1,int(2*us)));
    }
    int y=y0+titleH+zoomH+int(18*us);
    const double fs=0.43*us;
    auto line=[&](const std::string& text,cv::Scalar color=cv::Scalar(255,255,255)){
        putOutlinedText(output,text,{x0+margin,y},fs,color,std::max(1,int(1*us))); y+=lineH;};
    line(std::string("STATE: ")+stateName(r.state)+(showTarget?std::string("   SRC: ")+helipad::sourceName(r.target.source):""),stateColor(r.state));
    if(showTarget){
        if(r.guidanceValid){
            std::ostringstream s; s<<std::showpos<<std::fixed<<std::setprecision(1)<<"X: "<<r.guidance.errorX<<" px   Y: "<<r.guidance.errorY<<" px"; line(s.str());
            std::ostringstream t; t<<std::showpos<<std::fixed<<std::setprecision(2)<<"ROLL: "<<r.guidance.roll<<"   PITCH: "<<r.guidance.pitch; line(t.str());
        }else{line("GUIDANCE: HOLD",cv::Scalar(0,140,255));line("X/Y: prediction only");}
        std::ostringstream s3; s3<<std::noshowpos<<std::fixed<<std::setprecision(3)<<"H: "<<r.target.score<<"  TOPO: "<<r.target.topology<<"  RAIL: "<<r.target.railAvg; line(s3.str());
        std::ostringstream s4; s4<<std::fixed<<std::setprecision(3)<<"FLOW: "<<r.flowQuality<<"  IDMISS: "<<r.identityMisses<<"  LOST: "<<r.flowLostFrames; line(s4.str());
        std::ostringstream s5; s5<<std::fixed<<std::setprecision(0)<<"ID AGE: "<<r.identityAgeMs<<" ms   RING: "<<r.ringAgeMs<<" ms"; line(s5.str());
    }else{line("TARGET: ---");line("X/Y: ---");line("ROLL/PITCH: ---");line("H/FLOW: ---");}
    {std::ostringstream s;s<<"HMODE: "<<r.hProofMode<<"  METHOD: "<<r.acquisitionMethod<<"  CLASS: "<<r.targetClass;line(s.str());}
    {std::ostringstream s;s<<std::fixed<<std::setprecision(2)<<"GDINO: "<<(r.gdinoOn?"ON":"OFF")<<" "<<r.gdinoScore<<"  SAM2: "<<(r.sam2On?"ON":"OFF")<<" "<<r.samScore;line(s.str(),(r.gdinoOn&&r.sam2On)?cv::Scalar(0,255,0):cv::Scalar(0,140,255));}
    {std::ostringstream s;s<<std::fixed<<std::setprecision(1)<<"AI FPS: "<<r.aiFps<<"  AGE: "<<r.aiResultAgeMs<<" ms  REPL: "<<r.aiQueueReplaced;line(s.str());}
    {std::ostringstream s;s<<std::fixed<<std::setprecision(1)<<"PROC FPS: "<<procFps<<" / SRC "<<sourceFps<<"  PROC %: "<<processedPercent(procFps,sourceFps);line(s.str(),processedPercent(procFps,sourceFps)>=75.0?cv::Scalar(0,255,0):cv::Scalar(0,165,255));}
    {std::ostringstream s;s<<std::fixed<<std::setprecision(1)<<"TRACK CAP: "<<trackFps<<" FPS";line(s.str());}
}

cv::Mat composeOutput(const cv::Mat& original,const TrackResult& r,double procFps,
                      double sourceFps,double trackFps,int frameNo) {
    cv::Mat annotated=original.clone();
    annotateMain(annotated,r,procFps,sourceFps,trackFps,frameNo);
    drawTrackingInset(annotated,original,r,procFps,sourceFps,trackFps,0);
    return annotated;
}

VisionConfig makeConfig(int width,double inFps) {
    VisionConfig cfg;
    const double cadence=std::clamp(inFps/30.0,1.0,4.0);
    auto sc=[&](int base){return std::max(1,int(std::lround(base*cadence)));};
    cfg.max_lost_frames=sc(15);
    cfg.h_global_search_interval=sc(4);
    cfg.h_global_watch_interval=sc(30);
    cfg.h_global_confirm_interval=sc(2);
    cfg.h_dino_fallback_delay=sc(90);
    cfg.h_dino_fallback_interval=sc(90);
    cfg.h_verify_interval=sc(2);
    cfg.h_confirm_timeout=sc(18);
    cfg.h_far_search_interval=1;
    cfg.h_far_confirm_interval=1;
    cfg.h_far_grace_frames=sc(30);
    cfg.h_far_confirm_timeout=sc(32);
    cfg.universal_ring_interval=sc(4);
    cfg.ring_confirm_interval=sc(2);
    cfg.ring_confirm_timeout=sc(60);
    cfg.universal_ai_assist_delay=sc(30);
    cfg.ai_escalation_cooldown=sc(20);
    cfg.h_oblique_search_interval=sc(4);
    cfg.h_oblique_confirm_interval=sc(2);
    cfg.h_oblique_confirm_timeout=sc(24);

    cfg.h_far_search_interval=1;
    cfg.h_far_confirm_interval=1;
    cfg.h_far_candidates_per_scale=48;
    cfg.h_far_native_verify_limit=12;
    cfg.h_far_min_raw_span=5.5f;
    cfg.h_far_max_raw_span=120.0f;
    cfg.h_far_match_radius=55.0f;
    cfg.h_far_required_hits=3;
    cfg.h_far_x1=0.24f; cfg.h_far_x2=0.76f; cfg.h_far_y1=0.08f; cfg.h_far_y2=0.92f;

    cfg.h_far_search_interval=1;
    cfg.h_far_confirm_interval=1;
    cfg.h_far_required_hits=3;
    cfg.h_global_search_interval=2;
    cfg.h_oblique_search_interval=2;
    cfg.h_global_confirm_interval=1;
    cfg.locked_identity_period_ms=110;
    cfg.locked_identity_urgent_period_ms=45;
    cfg.identity_soft_lease_ms=320;
    cfg.identity_hard_lease_ms=720;
    cfg.identity_result_max_age_ms=460;
    cfg.locked_ring_period_ms=700;
    cfg.ring_label_lease_ms=1800;
    cfg.oblique_fast_required_hits=2;
    cfg.oblique_fast_min_gap_ms=55;
    cfg.oblique_fast_max_gap_ms=320;
    cfg.oblique_fast_verify_period_ms=90;
    cfg.oblique_fast_min_score=0.85f;
    cfg.oblique_fast_min_topology=0.93f;
    cfg.oblique_fast_min_rail_avg=0.50f;
    cfg.oblique_fast_min_rail=0.34f;
    cfg.oblique_fast_min_crossbar=0.42f;
    cfg.oblique_fast_min_void=0.55f;
    cfg.oblique_fast_min_contrast=82.0f;
    cfg.oblique_fast_authority_min=0.72f;
    cfg.confirm_competition_delay_ms=180;
    cfg.confirm_competition_period_ms=260;
    cfg.confirm_competition_margin=0.08f;
    cfg.confirming_ring_assist_period_ms=420;
    cfg.ring_fast_required_hits=2;
    cfg.ring_fast_required_h_hits=2;
    cfg.ring_fast_min_gap_ms=70;
    cfg.ring_fast_max_gap_ms=1800;
    cfg.ring_fast_verify_period_ms=140;
    cfg.ring_deep_confirm_period_ms=900;
    cfg.ring_deep_stale_after_fast_ms=1400;
    cfg.ring_h_core_scale=0.72f;
    cfg.ring_h_fallback_min_ms=160;
    cfg.ring_fast_min_pair=0.28f;

    if(width<=1500){
        const int s92=std::max(640,int(std::lround(0.92*width)));
        const int s80=std::max(560,int(std::lround(0.80*width)));
        cfg.h_far_scale_sides={width,std::min(width,s92),width,std::min(width,s80)};
    }else if(width<=2200){
        cfg.h_far_scale_sides={std::min(width,1700),std::min(width,1500),std::min(width,1300),std::min(width,1100)};
    }else cfg.h_far_scale_sides={1900,1650,1400,1150};
    cfg.h_far_candidates_per_scale=48;
    cfg.h_far_native_verify_limit=16;
    cfg.ring_confirm_interval=2;
    cfg.universal_ring_interval=6;
    return cfg;
}

class LatestFrame {
    mutable std::mutex mutex_;
    cv::Mat frame_;
    std::uint64_t seq_=0;
public:
    void put(const cv::Mat& f,std::uint64_t seq){std::lock_guard<std::mutex> l(mutex_);frame_=f;seq_=seq;}
    bool get(cv::Mat& f,std::uint64_t& seq) const {std::lock_guard<std::mutex> l(mutex_);if(frame_.empty())return false;f=frame_;seq=seq_;return true;}
    void clear(){std::lock_guard<std::mutex> l(mutex_);frame_.release();seq_=0;}
};

struct Telemetry {
    mutable std::mutex mutex;
    std::uint64_t seq=0;
    bool detActive=false, aiRequested=false, aiReady=false, aiActive=false, targetValid=false, streamReady=false;
    int x=0,y=0;
    double procFps=0.0, sourceFps=0.0;
    std::uint64_t frameNo=0;
    std::string state="OFF";
    std::string line() const {
        std::lock_guard<std::mutex> l(mutex);
        std::ostringstream s;
        s<<"STATE "<<seq<<" "<<(detActive?1:0)<<" "<<(aiRequested?1:0)<<" "<<(aiReady?1:0)<<" "<<(aiActive?1:0)
         <<" "<<(targetValid?1:0)<<" "<<x<<" "<<y<<" "<<std::fixed<<std::setprecision(1)<<procFps<<" "<<sourceFps
         <<" "<<frameNo<<" "<<state<<" "<<(streamReady?1:0)<<"\n";
        return s.str();
    }
};

class RtspServer {
    const Config& cfg_;
    LatestFrame& latest_;
    std::atomic<bool>& streamReady_;
    GMainLoop* loop_=nullptr;
    GstRTSPServer* server_=nullptr;
    guint sourceId_=0, maintenanceId_=0;
    std::thread glibThread_, pushThread_;
    std::atomic<bool> running_{false}, resetClients_{false};
    std::mutex mediaMutex_;
    GstElement* appsrc_=nullptr;
    GstRTSPMedia* media_=nullptr;

    static GstRTSPStatusCode preDescribe(GstRTSPClient*,GstRTSPContext*,gpointer data){
        auto* self=static_cast<RtspServer*>(data);
        return self->streamReady_.load()?GST_RTSP_STS_OK:GST_RTSP_STS_SERVICE_UNAVAILABLE;
    }
    static void connected(GstRTSPServer*,GstRTSPClient* client,gpointer data){
        g_signal_connect(client,"pre-describe-request",G_CALLBACK(preDescribe),data);
        GstRTSPConnection* conn=gst_rtsp_client_get_connection(client);
        if(conn){
            GSocket* sock=gst_rtsp_connection_get_write_socket(conn);
            if(sock){
                const int fd=g_socket_get_fd(sock); int one=1, bytes=64*1024;
                setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
                setsockopt(fd,SOL_SOCKET,SO_SNDBUF,&bytes,sizeof(bytes));
            }
        }
    }
    static void unprepared(GstRTSPMedia* m,gpointer data){
        auto* self=static_cast<RtspServer*>(data);
        std::lock_guard<std::mutex> l(self->mediaMutex_);
        if(m!=self->media_) return;
        if(self->appsrc_){gst_object_unref(self->appsrc_);self->appsrc_=nullptr;}
        if(self->media_){gst_object_unref(self->media_);self->media_=nullptr;}
    }
    static void configured(GstRTSPMediaFactory*,GstRTSPMedia* media,gpointer data){
        auto* self=static_cast<RtspServer*>(data);
        GstElement* root=gst_rtsp_media_get_element(media);
        GstElement* src=gst_bin_get_by_name_recurse_up(GST_BIN(root),"src");
        if(src){
            g_object_set(src,"max-buffers",guint64(1),"max-bytes",guint64(0),"max-time",guint64(0),
                         "leaky-type",GST_APP_LEAKY_TYPE_DOWNSTREAM,"block",FALSE,"do-timestamp",TRUE,
                         "min-latency",gint64(0),nullptr);
            std::lock_guard<std::mutex> l(self->mediaMutex_);
            if(self->appsrc_) gst_object_unref(self->appsrc_);
            if(self->media_) gst_object_unref(self->media_);
            self->appsrc_=src;
            self->media_=GST_RTSP_MEDIA(gst_object_ref(media));
        }
        g_signal_connect(media,"unprepared",G_CALLBACK(unprepared),self);
        gst_object_unref(root);
    }
    void closeClients(){
        if(!server_)return;
        GList* clients=gst_rtsp_server_client_filter(server_,nullptr,nullptr);
        for(GList* p=clients;p;p=p->next){auto* c=GST_RTSP_CLIENT(p->data);gst_rtsp_client_close(c);g_object_unref(c);}
        g_list_free(clients);
    }
    static gboolean maintenance(gpointer data){
        auto* self=static_cast<RtspServer*>(data);
        if(self->resetClients_.exchange(false)) self->closeClients();
        return G_SOURCE_CONTINUE;
    }
    void pushLoop(){
        const auto period=std::chrono::nanoseconds(1000000000LL/cfg_.streamFps);
        auto next=std::chrono::steady_clock::now();
        std::uint64_t last=0;
        while(running_){
            std::this_thread::sleep_until(next); next=std::chrono::steady_clock::now()+period;
            if(!streamReady_.load()) continue;
            cv::Mat full; std::uint64_t seq=0;
            if(!latest_.get(full,seq) || seq==last) continue;
            cv::Mat small; cv::resize(full,small,{cfg_.streamWidth,cfg_.streamHeight},0,0,cv::INTER_AREA);
            if(!small.isContinuous()) small=small.clone();
            GstElement* src=nullptr;
            {std::lock_guard<std::mutex> l(mediaMutex_);if(appsrc_)src=GST_ELEMENT(gst_object_ref(appsrc_));}
            if(!src) continue;
            const gsize bytes=small.total()*small.elemSize();
            GstBuffer* buffer=gst_buffer_new_allocate(nullptr,bytes,nullptr);
            GstMapInfo map{};
            if(buffer && gst_buffer_map(buffer,&map,GST_MAP_WRITE)){
                std::memcpy(map.data,small.data,bytes);gst_buffer_unmap(buffer,&map);
                GST_BUFFER_PTS(buffer)=GST_CLOCK_TIME_NONE;
                GST_BUFFER_DTS(buffer)=GST_CLOCK_TIME_NONE;
                GST_BUFFER_DURATION(buffer)=gst_util_uint64_scale_int(1,GST_SECOND,cfg_.streamFps);
                GstFlowReturn flow=gst_app_src_push_buffer(GST_APP_SRC(src),buffer);
                if(flow==GST_FLOW_OK) last=seq;
            }else if(buffer) gst_buffer_unref(buffer);
            gst_object_unref(src);
        }
    }
public:
    RtspServer(const Config& cfg,LatestFrame& latest,std::atomic<bool>& ready):cfg_(cfg),latest_(latest),streamReady_(ready){}
    ~RtspServer(){stop();}
    bool start(){
        gst_init(nullptr,nullptr);
        loop_=g_main_loop_new(nullptr,FALSE); server_=gst_rtsp_server_new();
        gst_rtsp_server_set_service(server_,std::to_string(cfg_.rtspPort).c_str());
        auto* mounts=gst_rtsp_server_get_mount_points(server_); auto* factory=gst_rtsp_media_factory_new();
        gst_rtsp_media_factory_set_shared(factory,TRUE); gst_rtsp_media_factory_set_latency(factory,0);
        gst_rtsp_media_factory_set_protocols(factory,GST_RTSP_LOWER_TRANS_TCP);
        const std::string pipe="( appsrc name=src is-live=true format=time do-timestamp=true block=false max-buffers=1 max-bytes=0 max-time=0 leaky-type=downstream "
            "caps=video/x-raw,format=BGR,width="+std::to_string(cfg_.streamWidth)+",height="+std::to_string(cfg_.streamHeight)+",framerate="+std::to_string(cfg_.streamFps)+"/1 "
            "! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
            "! videoconvert n-threads=2 ! video/x-raw,format=I420 "
            "! x264enc tune=zerolatency speed-preset=ultrafast threads=2 sliced-threads=true bitrate="+std::to_string(cfg_.bitrateKbps)+
            " vbv-buf-capacity=100 key-int-max=15 bframes=0 sync-lookahead=0 rc-lookahead=0 byte-stream=true "
            "! h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au "
            "! rtph264pay name=pay0 pt=96 config-interval=-1 mtu=1200 )";
        gst_rtsp_media_factory_set_launch(factory,pipe.c_str());
        g_signal_connect(factory,"media-configure",G_CALLBACK(configured),this);
        gst_rtsp_mount_points_add_factory(mounts,"/main.264",factory); g_object_unref(mounts);
        g_signal_connect(server_,"client-connected",G_CALLBACK(connected),this);
        sourceId_=gst_rtsp_server_attach(server_,nullptr);
        if(!sourceId_) return false;
        maintenanceId_=g_timeout_add(100,maintenance,this);
        running_=true; glibThread_=std::thread([this]{g_main_loop_run(loop_);});
        pushThread_=std::thread([this]{pushLoop();});
        return true;
    }
    void resetClients(){resetClients_=true;}
    void stop(){
        if(!running_.exchange(false)) return;
        resetClients_=true;
        if(pushThread_.joinable()) pushThread_.join();
        if(loop_) g_main_loop_quit(loop_);
        if(glibThread_.joinable()) glibThread_.join();
        {std::lock_guard<std::mutex> l(mediaMutex_);if(appsrc_){gst_object_unref(appsrc_);appsrc_=nullptr;}if(media_){gst_object_unref(media_);media_=nullptr;}}
        if(maintenanceId_) g_source_remove(maintenanceId_);
        if(sourceId_) g_source_remove(sourceId_);
        if(server_) g_object_unref(server_);
        if(loop_) g_main_loop_unref(loop_);
        server_=nullptr; loop_=nullptr; sourceId_=maintenanceId_=0;
    }
};

class ControlServer {
    const Config& cfg_;
    std::atomic<bool>& detRequest_;
    std::atomic<bool>& aiRequest_;
    Telemetry& telemetry_;
    std::condition_variable& wake_;
    std::mutex& wakeMutex_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    int listenFd_=-1;

    static bool sendAll(int fd,const std::string& s){
        const char* p=s.data(); size_t left=s.size();
        while(left){ssize_t n=::send(fd,p,left,MSG_NOSIGNAL);if(n<0){if(errno==EINTR)continue;return false;}p+=n;left-=size_t(n);}return true;
    }
    void handleLine(int fd,std::string line){
        while(!line.empty()&&(line.back()=='\r'||line.back()=='\n'))line.pop_back();
        std::istringstream in(line); std::string cmd; in>>cmd;
        if(cmd=="HELLO"){sendAll(fd,"HELLO 1 NEWSPACE_AIR_FRESH\n");sendAll(fd,telemetry_.line());}
        else if(cmd=="DET"){
            int v=0;in>>v;detRequest_.store(v!=0);if(!v)aiRequest_.store(false);wake_.notify_all();
            sendAll(fd,std::string("ACK DET ")+(v?"1\n":"0\n"));
        }else if(cmd=="AI"){
            int v=0;in>>v;
            if(v && !detRequest_.load()){aiRequest_.store(false);sendAll(fd,"ERR AI DETECTION_OFF\n");sendAll(fd,"ACK AI 0\n");}
            else{aiRequest_.store(v!=0);sendAll(fd,std::string("ACK AI ")+(v?"1\n":"0\n"));}
        }else if(cmd=="GET"){sendAll(fd,telemetry_.line());}
        else if(cmd=="PING"){sendAll(fd,"PONG\n");}
        else sendAll(fd,"ERR UNKNOWN_COMMAND\n");
    }
    void clientLoop(int fd){
        int one=1;setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
        sendAll(fd,"HELLO 1 NEWSPACE_AIR_FRESH\n");sendAll(fd,telemetry_.line());
        std::string buffer; auto nextState=std::chrono::steady_clock::now();
        while(running_ && !g_stop){
            pollfd pfd{fd,POLLIN,0}; int pr=::poll(&pfd,1,50);
            if(pr<0){if(errno==EINTR)continue;break;}
            if(pr>0 && (pfd.revents&(POLLERR|POLLHUP|POLLNVAL)))break;
            if(pr>0 && (pfd.revents&POLLIN)){
                char tmp[2048];ssize_t n=::recv(fd,tmp,sizeof(tmp),0);if(n<=0)break;buffer.append(tmp,size_t(n));
                size_t pos=0;while((pos=buffer.find('\n'))!=std::string::npos){std::string line=buffer.substr(0,pos);buffer.erase(0,pos+1);handleLine(fd,line);}
            }
            const auto now=std::chrono::steady_clock::now();
            if(now>=nextState){if(!sendAll(fd,telemetry_.line()))break;nextState=now+100ms;}
        }
        ::close(fd);
    }
    void loop(){
        listenFd_=::socket(AF_INET,SOCK_STREAM,0);if(listenFd_<0){perror("socket");return;}
        int one=1;setsockopt(listenFd_,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
        sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_ANY);a.sin_port=htons(cfg_.controlPort);
        if(::bind(listenFd_,reinterpret_cast<sockaddr*>(&a),sizeof(a))<0){perror("bind control");return;}
        if(::listen(listenFd_,1)<0){perror("listen");return;}
        std::cout<<"[CONTROL] TCP listening on "<<cfg_.controlPort<<"\n";
        while(running_ && !g_stop){pollfd pfd{listenFd_,POLLIN,0};int pr=::poll(&pfd,1,250);if(pr<=0)continue;int fd=::accept(listenFd_,nullptr,nullptr);if(fd>=0)clientLoop(fd);}
    }
public:
    ControlServer(const Config& c,std::atomic<bool>& d,std::atomic<bool>& a,Telemetry& t,std::condition_variable& w,std::mutex& wm)
        :cfg_(c),detRequest_(d),aiRequest_(a),telemetry_(t),wake_(w),wakeMutex_(wm){}
    ~ControlServer(){stop();}
    void start(){running_=true;thread_=std::thread([this]{loop();});}
    void stop(){if(!running_.exchange(false))return;if(listenFd_>=0){::shutdown(listenFd_,SHUT_RDWR);::close(listenFd_);listenFd_=-1;}if(thread_.joinable())thread_.join();}
};

// Local preview must NEVER own or block the vision/control thread.  Some Jetson
// X11/HighGUI combinations can stall inside namedWindow()/imshow().  Keep the
// exact final annotated frame available locally, but run all HighGUI calls in a
// detached latest-frame worker.  If X11 hangs, detection/RTSP/control continue.
class AsyncLocalDisplay {
    struct Shared {
        std::atomic<bool> stop{false};
        std::atomic<bool> enabled{true};
        std::mutex mutex;
        std::condition_variable wake;
        cv::Mat latest;
        std::uint64_t seq=0;
        std::uint64_t shown=0;
    };
    std::shared_ptr<Shared> shared_;
public:
    AsyncLocalDisplay(){
        shared_=std::make_shared<Shared>();
        auto st=shared_;
        std::thread([st]{
            try{
                cv::namedWindow("NEWSPACE AIR FINAL DETECTION",cv::WINDOW_NORMAL);
                cv::resizeWindow("NEWSPACE AIR FINAL DETECTION",1600,900);
                std::cout<<"[DISPLAY] async local preview READY\\n";
                while(!st->stop.load(std::memory_order_acquire)){
                    cv::Mat frame;
                    {
                        std::unique_lock<std::mutex> lock(st->mutex);
                        st->wake.wait_for(lock,25ms,[&]{
                            return st->stop.load(std::memory_order_acquire) || st->seq!=st->shown;
                        });
                        if(st->stop.load(std::memory_order_acquire)) break;
                        if(st->seq!=st->shown){
                            st->latest.copyTo(frame);
                            st->shown=st->seq;
                        }
                    }
                    if(!frame.empty()) cv::imshow("NEWSPACE AIR FINAL DETECTION",frame);
                    const int k=cv::waitKey(1)&0xff;
                    if(k==27 || k=='q' || k=='Q'){
                        std::cout<<"[DISPLAY] preview closed; detection/stream continue\\n";
                        break;
                    }
                }
                try{cv::destroyWindow("NEWSPACE AIR FINAL DETECTION");}catch(...){}
            }catch(const cv::Exception& e){
                std::cerr<<"[DISPLAY] local preview unavailable: "<<e.what()<<"; continuing headless\\n";
            }catch(...){
                std::cerr<<"[DISPLAY] local preview unavailable; continuing headless\\n";
            }
            st->enabled.store(false,std::memory_order_release);
        }).detach();
        std::cout<<"[DISPLAY] async preview requested; X11 cannot block vision/control\\n";
    }
    ~AsyncLocalDisplay(){stop();}
    void submit(const cv::Mat& frame){
        auto st=shared_;
        if(!st || !st->enabled.load(std::memory_order_acquire) || frame.empty()) return;
        {
            std::lock_guard<std::mutex> lock(st->mutex);
            frame.copyTo(st->latest);
            ++st->seq;
        }
        st->wake.notify_one();
    }
    void stop(){
        auto st=shared_;
        if(!st) return;
        st->stop.store(true,std::memory_order_release);
        st->enabled.store(false,std::memory_order_release);
        st->wake.notify_all();
        shared_.reset();
    }
};

bool localDisplayWanted(const Config& c){
    // PORTABLE_R2: never call HighGUI/X11 from publisher_node.  On Jetson,
    // an Xlib fatal error terminates the whole process even if HighGUI runs in
    // a detached thread.  Local diagnostic display is provided by
    // local_viewer.sh as a separate localhost RTSP client.
    (void)c;
    return false;
}

} // namespace

int main(int argc,char** argv){
    std::signal(SIGINT,onSignal);std::signal(SIGTERM,onSignal);
    Config cfg;
    try{if(!parseArgs(argc,argv,cfg))return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";usage(argv[0]);return 2;}

    std::atomic<bool> detRequest{false},aiRequest{false},detActive{false},streamReady{false};
    LatestFrame latest; Telemetry telemetry; std::condition_variable wake; std::mutex wakeMutex;
    RtspServer rtsp(cfg,latest,streamReady);
    if(!rtsp.start()){std::cerr<<"Failed to start RTSP server on "<<cfg.rtspPort<<"\n";return 3;}
    ControlServer control(cfg,detRequest,aiRequest,telemetry,wake,wakeMutex); control.start();
    std::cout<<"[AIR] publisher_node ready. Detection starts only after DET 1.\n";
    std::cout<<"[AIR] RTSP rtsp://<jetson>:"<<cfg.rtspPort<<"/main.264 (TCP only)\n";

    const bool preview=localDisplayWanted(cfg);
    std::unique_ptr<AsyncLocalDisplay> localDisplay;
    while(!g_stop){
        if(!detRequest.load()){
            std::unique_lock<std::mutex> l(wakeMutex);wake.wait_for(l,250ms,[&]{return g_stop||detRequest.load();});continue;
        }
        std::cout<<"[VISION] DET ON: opening CSI and detector\n";
        cv::VideoCapture cap(mipiPipeline(cfg),cv::CAP_GSTREAMER);
        if(!cap.isOpened()){
            std::cerr<<"[VISION] CSI open failed; retrying while DET remains ON\n";
            {std::lock_guard<std::mutex> l(telemetry.mutex);telemetry.detActive=false;telemetry.state="CAMERA_ERROR";++telemetry.seq;}
            std::this_thread::sleep_for(1s);continue;
        }
        auto tracker=std::make_unique<HelipadTracker>(makeConfig(cfg.width,cfg.fps));
        bool aiReady=false;
        if(!cfg.dinoPlan.empty()&&!cfg.samPlan.empty()&&std::filesystem::exists(cfg.dinoPlan)&&std::filesystem::exists(cfg.samPlan)){
            std::cout<<"[AI] Loading TensorRT engines once for this Detection session...\n";
            aiReady=tracker->setAiEngines(cfg.dinoPlan,cfg.samPlan);
            std::cout<<"[AI] engines "<<(aiReady?"READY":"UNAVAILABLE")<<"\n";
        }
        tracker->setAiEnabled(false);
        detActive=true;
        {std::lock_guard<std::mutex> l(telemetry.mutex);telemetry.detActive=true;telemetry.aiReady=aiReady;telemetry.sourceFps=cfg.fps;telemetry.streamReady=false;telemetry.state="SEARCHING";++telemetry.seq;}
        std::uint64_t frameNo=0;int winFrames=0;double procFps=0.0,trackCap=0.0,trackSeconds=0.0;
        auto winStart=std::chrono::steady_clock::now(); bool lastAi=false;
        if(preview) localDisplay=std::make_unique<AsyncLocalDisplay>();
        std::cout<<"[VISION] capture loop START; waiting for first CSI frame\n";
        while(!g_stop && detRequest.load()){
            cv::Mat frame;if(!cap.read(frame)||frame.empty()){std::cerr<<"[VISION] CSI frame read failed\n";break;}
            ++frameNo;++winFrames;
            if(frameNo==1) std::cout<<"[VISION] first CSI frame RECEIVED\n";
            const bool wantAi=aiReady && aiRequest.load();
            if(wantAi!=lastAi){tracker->setAiEnabled(wantAi);lastAi=wantAi;std::cout<<"[AI] "<<(wantAi?"ON":"OFF")<<"\n";}
            const auto t0=std::chrono::steady_clock::now();
            TrackResult r=tracker->process(frame);
            const auto t1=std::chrono::steady_clock::now();
            if(frameNo==1) std::cout<<"[VISION] first detector result READY\n";
            trackSeconds+=std::chrono::duration<double>(t1-t0).count();
            const double wall=std::chrono::duration<double>(t1-winStart).count();
            if(wall>=0.5){procFps=winFrames/wall;trackCap=winFrames/std::max(1e-6,trackSeconds);winFrames=0;trackSeconds=0;winStart=t1;}
            cv::Mat finalFrame=composeOutput(frame,r,procFps,cfg.fps,trackCap,int(frameNo));
            if(frameNo==1) std::cout<<"[VISION] first FINAL annotated frame COMPOSED\n";
            latest.put(finalFrame,frameNo);
            if(!streamReady.exchange(true)){
                {
                    std::lock_guard<std::mutex> l(telemetry.mutex);
                    telemetry.streamReady=true;
                    ++telemetry.seq;
                }
                std::cout<<"[RTSP] first final frame READY; remote RTSP may connect now\n";
                rtsp.resetClients();
            }
            {
                std::lock_guard<std::mutex> l(telemetry.mutex);++telemetry.seq;telemetry.detActive=true;telemetry.aiRequested=aiRequest.load();telemetry.aiReady=aiReady;
                telemetry.aiActive=r.gdinoOn&&r.sam2On;telemetry.targetValid=r.guidanceValid;telemetry.x=r.guidanceValid?int(std::lround(r.guidance.errorX)):0;
                telemetry.y=r.guidanceValid?int(std::lround(r.guidance.errorY)):0;telemetry.procFps=procFps;telemetry.sourceFps=cfg.fps;telemetry.frameNo=frameNo;telemetry.state=stateName(r.state);
            }
            // Local preview is diagnostic only. At 60-Hz camera input submit at
            // most 30 FPS to a one-slot worker; it can never back-pressure vision.
            if(preview && localDisplay && (frameNo==1 || (frameNo%2)==0))
                localDisplay->submit(finalFrame);
        }
        streamReady=false;
        tracker->setAiEnabled(false);tracker.reset();cap.release();detActive=false;aiRequest=false;latest.clear();rtsp.resetClients();
        if(localDisplay){localDisplay->stop();localDisplay.reset();}
        {std::lock_guard<std::mutex> l(telemetry.mutex);++telemetry.seq;telemetry.detActive=false;telemetry.aiRequested=false;telemetry.aiReady=false;telemetry.aiActive=false;telemetry.targetValid=false;telemetry.streamReady=false;telemetry.x=telemetry.y=0;telemetry.procFps=0;telemetry.frameNo=0;telemetry.state="OFF";}
        std::cout<<"[VISION] DET OFF: CSI, detector, AI and RTSP media stopped\n";
    }
    detRequest=false;aiRequest=false;detActive=false;streamReady=false;latest.clear();rtsp.resetClients();control.stop();rtsp.stop();
    if(localDisplay){localDisplay->stop();localDisplay.reset();}
    return 0;
}
