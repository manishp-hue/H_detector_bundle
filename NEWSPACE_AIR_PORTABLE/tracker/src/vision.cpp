#include <atomic>
#include <iostream>
#include <chrono>
#include <future>
#include "vision.hpp"
#include "corelock05_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <utility>
#include <numeric>
#include <stdexcept>

namespace helipad {
namespace {
constexpr bool kRingProfiling = false;


float percentile(std::vector<float> v, float p) {
    if (v.empty()) return 0.0f;
    p = std::clamp(p, 0.0f, 100.0f);
    size_t idx = static_cast<size_t>(std::round((p / 100.0f) * float(v.size() - 1)));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

float median(std::vector<float> v) {
    return percentile(std::move(v), 50.0f);
}

float angleDelta180(float a, float b) {
    float d = std::fmod(std::fabs(a - b), 180.0f);
    return std::min(d, 180.0f - d);
}

std::array<cv::Point2f, 4> orderQuad(const cv::Point2f pts[4]) {
    std::array<cv::Point2f, 4> p{pts[0], pts[1], pts[2], pts[3]};
    auto sum = [](const cv::Point2f& q) { return q.x + q.y; };
    auto diff = [](const cv::Point2f& q) { return q.y - q.x; };
    auto tl = *std::min_element(p.begin(), p.end(), [&](auto& a, auto& b){ return sum(a) < sum(b); });
    auto tr = *std::min_element(p.begin(), p.end(), [&](auto& a, auto& b){ return diff(a) < diff(b); });
    auto br = *std::max_element(p.begin(), p.end(), [&](auto& a, auto& b){ return sum(a) < sum(b); });
    auto bl = *std::max_element(p.begin(), p.end(), [&](auto& a, auto& b){ return diff(a) < diff(b); });
    return {tl, tr, br, bl};
}

float zoneMean01(const cv::Mat& m, float x0, float x1, float y0, float y1) {
    if (m.empty()) return 0.0f;
    int xa = std::clamp(int(x0 * m.cols), 0, std::max(0, m.cols - 1));
    int xb = std::clamp(int(x1 * m.cols), xa + 1, m.cols);
    int ya = std::clamp(int(y0 * m.rows), 0, std::max(0, m.rows - 1));
    int yb = std::clamp(int(y1 * m.rows), ya + 1, m.rows);
    cv::Scalar s = cv::mean(m(cv::Rect(xa, ya, xb - xa, yb - ya)));
    float v = static_cast<float>(s[0]);
    if (m.depth() == CV_8U) v /= 255.0f;
    return v;
}

float hIdentityThreshold(HMode mode) {
    switch (mode) {
        case HMode::Acquire: return 0.78f;
        case HMode::Confirm: return 0.72f;
        case HMode::Recovery: return 0.70f;
        case HMode::Track: return 0.68f;
        case HMode::Far: return 0.0f;
    }
    return 0.78f;
}

cv::Rect2f rectFromCorners(const std::vector<cv::Point2f>& pts, int width, int height) {
    if (pts.empty()) return {};
    float x1 = pts[0].x, y1 = pts[0].y, x2 = pts[0].x, y2 = pts[0].y;
    for (const auto& p : pts) {
        x1 = std::min(x1, p.x); y1 = std::min(y1, p.y);
        x2 = std::max(x2, p.x); y2 = std::max(y2, p.y);
    }
    return clampRect({x1, y1, std::max(1.0f, x2-x1), std::max(1.0f, y2-y1)}, width, height);
}



// ============================================================
// V15.54 CORELOCK05 DUAL-PROOF GEOMETRY
//
// Aryan's existing HTopologyMetrics remains the positive detector authority.
// PureH_CoreLock_05 is added as an independent negative structural guard and
// FAR micro-topology proof.  No tracker/AI/lease/guidance state lives here.
// ============================================================
constexpr int kCoreLockCanon = 64;
constexpr float kCoreLockStrictMajorPx = 24.0f;

struct CoreLock05Evidence {
    helipad_corelock05::FarTopologyDecision signature;
    helipad_corelock05::IntegrityResult integrity;
    helipad_corelock05::FarTopologyMetrics topology;
};

// V15.55: CoreLock05 scans many small canonical zones. Use one integral image
// per candidate so each zone mean is O(1) instead of repeatedly constructing
// ROIs and calling cv::mean(). The result is numerically equivalent for a
// binary 0/255 mask.
float coreZoneMean(const cv::Mat& integralMask, double x1, double y1, double x2, double y2) {
    const int ix1=std::clamp(int(std::floor(x1*kCoreLockCanon)),0,kCoreLockCanon-1);
    const int iy1=std::clamp(int(std::floor(y1*kCoreLockCanon)),0,kCoreLockCanon-1);
    const int ix2=std::clamp(int(std::ceil(x2*kCoreLockCanon)),ix1+1,kCoreLockCanon);
    const int iy2=std::clamp(int(std::ceil(y2*kCoreLockCanon)),iy1+1,kCoreLockCanon);
    const int sum=
        integralMask.at<int>(iy2,ix2)-integralMask.at<int>(iy1,ix2)-
        integralMask.at<int>(iy2,ix1)+integralMask.at<int>(iy1,ix1);
    const int area=std::max(1,(ix2-ix1)*(iy2-iy1));
    return float(sum)/(255.0f*float(area));
}

CoreLock05Evidence evaluateCoreLock05One(const cv::Mat& input, float sourceMajorPx) {
    CoreLock05Evidence out;
    if(input.empty()) {
        out.signature.kind=helipad_corelock05::FarSignatureKind::Reject;
        out.integrity.pass=false;
        out.integrity.score=0.0;
        return out;
    }

    cv::Mat mask;
    cv::resize(input,mask,cv::Size(kCoreLockCanon,kCoreLockCanon),0,0,cv::INTER_NEAREST);
    cv::threshold(mask,mask,1,255,cv::THRESH_BINARY);
    cv::Mat integralMask;
    cv::integral(mask,integralMask,CV_32S);
    std::vector<cv::Point> points;
    cv::findNonZero(mask,points);
    if(points.size()<5) {
        out.signature.kind=helipad_corelock05::FarSignatureKind::Reject;
        out.integrity.pass=false;
        out.integrity.score=0.0;
        return out;
    }

    auto stemSupport=[](double v){
        return helipad_corelock05::clamp01((v-0.014)/0.155);
    };

    double bestBridgeRaw=-1.0;
    double bestBridgeY=0.5;
    for(int yi=14;yi<=50;++yi) {
        const double y=double(yi)/kCoreLockCanon;
        const double stripe=coreZoneMean(integralMask,0.20,y-0.035,0.80,y+0.035);
        const double center=coreZoneMean(integralMask,0.40,y-0.045,0.60,y+0.045);
        const double sideL=coreZoneMean(integralMask,0.08,y-0.050,0.40,y+0.050);
        const double sideR=coreZoneMean(integralMask,0.60,y-0.050,0.92,y+0.050);
        const double raw=0.50*center+0.30*stripe+0.10*sideL+0.10*sideR;
        if(raw>bestBridgeRaw){bestBridgeRaw=raw;bestBridgeY=y;}
    }

    const double topEnd=std::max(0.15,bestBridgeY-0.08);
    const double bottomStart=std::min(0.85,bestBridgeY+0.08);
    const double leftTop=stemSupport(coreZoneMean(integralMask,0.08,0.06,0.40,topEnd));
    const double rightTop=stemSupport(coreZoneMean(integralMask,0.60,0.06,0.92,topEnd));
    const double leftBottom=stemSupport(coreZoneMean(integralMask,0.08,bottomStart,0.40,0.94));
    const double rightBottom=stemSupport(coreZoneMean(integralMask,0.60,bottomStart,0.92,0.94));
    const double bridge=helipad_corelock05::clamp01((bestBridgeRaw-0.025)/0.30);
    const double topGapOcc=coreZoneMean(integralMask,0.43,0.06,0.57,topEnd);
    const double bottomGapOcc=coreZoneMean(integralMask,0.43,bottomStart,0.57,0.94);
    const double gapTop=helipad_corelock05::clamp01(1.0-topGapOcc/0.36);
    const double gapBottom=helipad_corelock05::clamp01(1.0-bottomGapOcc/0.36);
    const double leftAvg=0.5*(leftTop+leftBottom);
    const double rightAvg=0.5*(rightTop+rightBottom);
    const double balance=helipad_corelock05::clamp01(
        1.0-std::abs(leftAvg-rightAvg)/std::max(0.18,std::max(leftAvg,rightAvg)));

    const cv::Rect fgBox=cv::boundingRect(points);
    const double fill=double(points.size())/std::max(1.0,double(fgBox.area()));
    cv::Mat labels,stats,centroids;
    const int ncomp=cv::connectedComponentsWithStats(mask,labels,stats,centroids,8,CV_32S);
    int largest=0;
    for(int label=1;label<ncomp;++label)
        largest=std::max(largest,stats.at<int>(label,cv::CC_STAT_AREA));
    const double largestFraction=double(largest)/std::max(1.0,double(points.size()));

    int twoSideRows=0,testedRows=0;
    for(int y=int(0.08*kCoreLockCanon);y<int(0.92*kCoreLockCanon);++y) {
        const cv::Mat row=mask.row(y);
        const int lp=cv::countNonZero(row.colRange(int(0.08*kCoreLockCanon),int(0.42*kCoreLockCanon)));
        const int rp=cv::countNonZero(row.colRange(int(0.58*kCoreLockCanon),int(0.92*kCoreLockCanon)));
        const int cp=cv::countNonZero(row.colRange(int(0.43*kCoreLockCanon),int(0.57*kCoreLockCanon)));
        if(lp>=2&&rp>=2&&cp<=3)++twoSideRows;
        ++testedRows;
    }

    out.topology.leftTop=leftTop;
    out.topology.leftBottom=leftBottom;
    out.topology.rightTop=rightTop;
    out.topology.rightBottom=rightBottom;
    out.topology.bridgeStrength=bridge;
    out.topology.bridgeY=bestBridgeY;
    out.topology.gapTop=gapTop;
    out.topology.gapBottom=gapBottom;
    out.topology.leftRightBalance=balance;
    out.topology.blobCompactness=helipad_corelock05::clamp01(fill);
    out.topology.largestComponentFraction=helipad_corelock05::clamp01(largestFraction);
    out.topology.twoLegRowContinuity=testedRows?double(twoSideRows)/testedRows:0.0;
    out.signature=helipad_corelock05::classifyFarTopology(out.topology);

    helipad_corelock05::IntegrityMetrics im;
    im.resolved=sourceMajorPx>=kCoreLockStrictMajorPx;
    if(!im.resolved) {
        out.integrity=helipad_corelock05::evaluateIntegrity(im);
        return out;
    }

    im.fillRatio=fill;
    const double lt=coreZoneMean(integralMask,0.12,0.08,0.36,0.40);
    const double lb=coreZoneMean(integralMask,0.12,0.60,0.36,0.92);
    const double rt=coreZoneMean(integralMask,0.64,0.08,0.88,0.40);
    const double rb=coreZoneMean(integralMask,0.64,0.60,0.88,0.92);
    const double ib=coreZoneMean(integralMask,0.34,0.40,0.66,0.60);
    const double tg=coreZoneMean(integralMask,0.43,0.08,0.57,0.34);
    const double bg=coreZoneMean(integralMask,0.43,0.66,0.57,0.92);
    auto support=[](double v){return helipad_corelock05::clamp01((v-0.035)/0.22);};
    std::array<double,4> stems{{support(lt),support(lb),support(rt),support(rb)}};
    std::sort(stems.begin(),stems.end());
    im.threeOfFourStemSupport=stems[1];
    im.bothSidesSupport=std::min(std::max(support(lt),support(lb)),std::max(support(rt),support(rb)));
    im.bridgeSupport=helipad_corelock05::clamp01((ib-0.035)/0.27);
    im.centerGapScore=helipad_corelock05::clamp01(1.0-(tg+bg)/0.58);

    cv::Mat profile;
    cv::morphologyEx(mask,profile,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3)));
    auto legRows=[&](double y1,double y2){
        const int ys=std::clamp(int(y1*kCoreLockCanon),0,kCoreLockCanon-1);
        const int ye=std::clamp(int(y2*kCoreLockCanon),ys+1,kCoreLockCanon);
        int good=0,total=0;
        for(int y=ys;y<ye;++y){
            const cv::Mat row=profile.row(y);
            const int l=cv::countNonZero(row.colRange(int(.08*kCoreLockCanon),int(.42*kCoreLockCanon)));
            const int r=cv::countNonZero(row.colRange(int(.58*kCoreLockCanon),int(.92*kCoreLockCanon)));
            const int c=cv::countNonZero(row.colRange(int(.43*kCoreLockCanon),int(.57*kCoreLockCanon)));
            if(l>=2&&r>=2&&c<=2)++good;
            ++total;
        }
        return total?double(good)/total:0.0;
    };
    im.legRowContinuity=std::min(legRows(.08,.38),legRows(.62,.92));

    cv::Mat bridgeProfile;
    cv::morphologyEx(profile,bridgeProfile,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,1)));
    int bridgeGood=0,bridgeTotal=0;
    for(int y=int(.36*kCoreLockCanon);y<int(.64*kCoreLockCanon);++y){
        const cv::Mat row=bridgeProfile.row(y);
        std::vector<cv::Point> rp;cv::findNonZero(row,rp);
        bool spans=false;
        if(!rp.empty()){
            int minX=kCoreLockCanon,maxX=-1;
            for(const auto&p:rp){minX=std::min(minX,p.x);maxX=std::max(maxX,p.x);}
            const int centerSupport=cv::countNonZero(row.colRange(int(.43*kCoreLockCanon),int(.57*kCoreLockCanon)));
            spans=minX<int(.35*kCoreLockCanon)&&maxX>int(.65*kCoreLockCanon)&&centerSupport>=2;
        }
        if(spans)++bridgeGood;
        ++bridgeTotal;
    }
    im.bridgeRowContinuity=bridgeTotal?double(bridgeGood)/bridgeTotal:0.0;

    cv::Mat connected;
    cv::morphologyEx(mask,connected,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3)));
    cv::Mat clabels,cstats,ccentroids;
    const int cc=cv::connectedComponentsWithStats(connected,clabels,cstats,ccentroids,8,CV_32S);
    const int total=std::max(1,cv::countNonZero(connected));
    int clargest=0;bool bridgeConnected=false;
    for(int label=1;label<cc;++label){
        const int a=cstats.at<int>(label,cv::CC_STAT_AREA);
        clargest=std::max(clargest,a);
        if(double(a)/total<0.42)continue;
        const int x=cstats.at<int>(label,cv::CC_STAT_LEFT);
        const int w=cstats.at<int>(label,cv::CC_STAT_WIDTH);
        const int h=cstats.at<int>(label,cv::CC_STAT_HEIGHT);
        cv::Mat component=clabels==label;
        const int ls=cv::countNonZero(component(cv::Rect(int(.08*kCoreLockCanon),int(.12*kCoreLockCanon),int(.35*kCoreLockCanon),int(.76*kCoreLockCanon))));
        const int rs=cv::countNonZero(component(cv::Rect(int(.57*kCoreLockCanon),int(.12*kCoreLockCanon),int(.35*kCoreLockCanon),int(.76*kCoreLockCanon))));
        const int bs=cv::countNonZero(component(cv::Rect(int(.30*kCoreLockCanon),int(.36*kCoreLockCanon),int(.40*kCoreLockCanon),int(.28*kCoreLockCanon))));
        if(x<int(.43*kCoreLockCanon)&&x+w>int(.57*kCoreLockCanon)&&h>int(.42*kCoreLockCanon)&&ls>=3&&rs>=3&&bs>=3)
            bridgeConnected=true;
    }
    im.spanningComponentFraction=double(clargest)/total;
    im.bridgeConnectivity=bridgeConnected;
    out.integrity=helipad_corelock05::evaluateIntegrity(im);
    return out;
}

CoreLock05Evidence evaluateCoreLock05(const cv::Mat& normalizedMask,float sourceMajorPx){
    auto a=evaluateCoreLock05One(normalizedMask,sourceMajorPx);

    // V15.55 FAST ORIENTATION EXIT. PureH/CoreLock05 only needs the second
    // 90-degree interpretation when the first interpretation is not already a
    // very strong H. Marginal candidates still receive the full two-way test.
    // This preserves difficult/perspective cases while removing a large amount
    // of duplicate canonical work on clear H frames.
    if(sourceMajorPx>=kCoreLockStrictMajorPx){
        if(a.integrity.hardChecked && a.integrity.pass && a.integrity.score>=0.78)
            return a;
    }else if(a.signature.kind==helipad_corelock05::FarSignatureKind::Strong &&
             a.signature.score>=0.70){
        return a;
    }

    cv::Mat rot;cv::rotate(normalizedMask,rot,cv::ROTATE_90_CLOCKWISE);
    auto b=evaluateCoreLock05One(rot,sourceMajorPx);
    auto sigRank=[](helipad_corelock05::FarSignatureKind k){
        return k==helipad_corelock05::FarSignatureKind::Strong?2:
               (k==helipad_corelock05::FarSignatureKind::Ambiguous?1:0);
    };
    auto rank=[&](const CoreLock05Evidence&e){
        const double integrityPriority=(sourceMajorPx>=kCoreLockStrictMajorPx&&e.integrity.pass)?100.0:0.0;
        return integrityPriority+10.0*sigRank(e.signature.kind)+e.signature.score+0.25*e.integrity.score;
    };
    return rank(a)>=rank(b)?a:b;
}

float coreLockQuality(const CoreLock05Evidence& e,float sourceMajorPx){
    if(sourceMajorPx>=kCoreLockStrictMajorPx){
        return float(std::clamp(0.78+0.22*e.integrity.score,0.0,1.0));
    }
    if(e.signature.kind==helipad_corelock05::FarSignatureKind::Strong)
        return float(std::clamp(0.92+0.08*e.signature.score,0.0,1.0));
    if(e.signature.kind==helipad_corelock05::FarSignatureKind::Ambiguous)
        return float(std::clamp(0.78+0.16*e.signature.score,0.0,1.0));
    return 0.70f;
}

// CoreLock_05 uses a 5% local clearance halo. Aryan's ring lane must remain
// legal, so this merged detector uses the same locality as a SOFT quality term;
// it never independently rejects a ring-backed H.
float localHaloQuietScore(const cv::Mat& binaryMask,const cv::Rect& body){
    if(binaryMask.empty()||body.area()<=0)return 1.0f;
    const int halo=std::max(1,int(std::ceil(.05f*std::max(body.width,body.height))));
    cv::Rect outer(body.x-halo,body.y-halo,body.width+2*halo,body.height+2*halo);
    outer&=cv::Rect(0,0,binaryMask.cols,binaryMask.rows);
    const int annulus=std::max(1,outer.area()-body.area());
    const int outerNz=cv::countNonZero(binaryMask(outer));
    const int bodyNz=cv::countNonZero(binaryMask(body));
    const float ratio=float(std::max(0,outerNz-bodyNz))/annulus;
    return std::clamp(1.0f-ratio/.45f,0.0f,1.0f);
}

} // namespace

cv::Rect2f clampRect(const cv::Rect2f& r, int width, int height) {
    if (width <= 1 || height <= 1) return {};
    float x1 = std::clamp(r.x, 0.0f, float(width - 1));
    float y1 = std::clamp(r.y, 0.0f, float(height - 1));
    float x2 = std::clamp(r.x + r.width, x1 + 1.0f, float(width));
    float y2 = std::clamp(r.y + r.height, y1 + 1.0f, float(height));
    return {x1, y1, x2 - x1, y2 - y1};
}

cv::Rect2f expandRect(const cv::Rect2f& r, float factor, int width, int height) {
    cv::Point2f c = rectCenter(r);
    float nw = std::max(2.0f, r.width * factor);
    float nh = std::max(2.0f, r.height * factor);
    return clampRect({c.x - 0.5f*nw, c.y - 0.5f*nh, nw, nh}, width, height);
}

cv::Rect2f moveRect(const cv::Rect2f& r, float dx, float dy) {
    return {r.x + dx, r.y + dy, r.width, r.height};
}

float rectSpan(const cv::Rect2f& r) { return std::max(r.width, r.height); }
cv::Point2f rectCenter(const cv::Rect2f& r) { return {r.x + 0.5f*r.width, r.y + 0.5f*r.height}; }

HGeometryDetector::HGeometryDetector(VisionConfig cfg) : cfg_(std::move(cfg)) {}

cv::Rect2f HGeometryDetector::searchBox(const cv::Mat& bgr,
                                         std::optional<cv::Rect2f> anchorBox,
                                         std::optional<cv::Rect2f> predictedBox,
                                         HMode mode) const {
    const int width = bgr.cols, height = bgr.rows;
    std::optional<cv::Rect2f> ref = predictedBox ? predictedBox : anchorBox;
    if (!ref) return {0, 0, float(width), float(height)};

    cv::Rect2f a = clampRect(*ref, width, height);
    cv::Point2f c = rectCenter(a);
    float span = rectSpan(a);
    float rw = 0.0f, rh = 0.0f;
    if (mode == HMode::Acquire) {
        rw = std::max({420.0f, width * 0.46f, span * 15.0f});
        rh = std::max({320.0f, height * 0.50f, span * 15.0f});
    } else if (mode == HMode::Recovery) {
        rw = std::max(320.0f, span * 8.0f);
        rh = std::max(260.0f, span * 8.0f);
    } else if (mode == HMode::Confirm) {
        rw = std::max(160.0f, span * 5.0f);
        rh = std::max(140.0f, span * 5.0f);
    } else {
        rw = std::max(140.0f, span * 4.2f);
        rh = std::max(120.0f, span * 4.2f);
    }
    rw = std::min(rw, float(width));
    rh = std::min(rh, float(height));
    return clampRect({c.x - 0.5f*rw, c.y - 0.5f*rh, rw, rh}, width, height);
}

cv::Mat HGeometryDetector::contrastMask(const cv::Mat& bgrCrop,
                                         std::optional<float> expectedSpan,
                                         bool globalMode) const {
    cv::Mat gray;
    cv::cvtColor(bgrCrop, gray, cv::COLOR_BGR2GRAY);
    const double sigmaSmall = 3.0;
    double sigmaLarge = globalMode ? 11.0 : 8.0;
    if (expectedSpan) sigmaLarge = std::clamp(double(*expectedSpan) * 0.18, 6.0, 30.0);

    cv::Mat b1, b2;
    cv::GaussianBlur(gray, b1, {0,0}, sigmaSmall);
    cv::GaussianBlur(gray, b2, {0,0}, sigmaLarge);

    cv::Mat g16, b116, b216, d1, d2;
    gray.convertTo(g16, CV_16S); b1.convertTo(b116, CV_16S); b2.convertTo(b216, CV_16S);
    cv::subtract(g16, b116, d1); cv::subtract(g16, b216, d2);

    cv::Mat gray92, gray188, d1_8, d2_13, d2_7, relative, bright, tmp, mask;
    cv::compare(gray, cv::Scalar(92), gray92, cv::CMP_GE);
    cv::compare(gray, cv::Scalar(188), gray188, cv::CMP_GE);
    cv::compare(d1, cv::Scalar(8), d1_8, cv::CMP_GE);
    cv::compare(d2, cv::Scalar(13), d2_13, cv::CMP_GE);
    cv::compare(d2, cv::Scalar(7), d2_7, cv::CMP_GE);
    cv::bitwise_or(d1_8, d2_13, tmp);
    cv::bitwise_and(gray92, tmp, relative);
    cv::bitwise_and(gray188, d2_7, bright);
    cv::bitwise_or(relative, bright, mask);

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat::ones(2,2,CV_8U));
    int k = (expectedSpan && *expectedSpan >= 90.0f) ? 4 : 3;
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat::ones(k,k,CV_8U));
    return mask;
}

cv::Mat HGeometryDetector::farContrastMask(const cv::Mat& bgrCrop) const {
    cv::Mat gray;
    cv::cvtColor(bgrCrop, gray, cv::COLOR_BGR2GRAY);
    cv::Mat b1, b2;
    cv::GaussianBlur(gray, b1, {0,0}, 1.6);
    cv::GaussianBlur(gray, b2, {0,0}, 6.0);
    cv::Mat g16,b116,b216,d1,d2;
    gray.convertTo(g16,CV_16S); b1.convertTo(b116,CV_16S); b2.convertTo(b216,CV_16S);
    cv::subtract(g16,b116,d1); cv::subtract(g16,b216,d2);
    cv::Mat gray88,gray175,d1_5,d2_10,d2_5,tmp,relative,bright,mask;
    cv::compare(gray,cv::Scalar(88),gray88,cv::CMP_GE); cv::compare(gray,cv::Scalar(175),gray175,cv::CMP_GE);
    cv::compare(d1,cv::Scalar(5),d1_5,cv::CMP_GE); cv::compare(d2,cv::Scalar(10),d2_10,cv::CMP_GE); cv::compare(d2,cv::Scalar(5),d2_5,cv::CMP_GE);
    cv::bitwise_or(d1_5,d2_10,tmp); cv::bitwise_and(gray88,tmp,relative); cv::bitwise_and(gray175,d2_5,bright);
    cv::bitwise_or(relative,bright,mask);
    cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,cv::Mat::ones(2,2,CV_8U));
    return mask;
}

cv::Mat HGeometryDetector::normalizedComponentMask(const std::vector<cv::Point>& contour) const {
    if (contour.size() < 5) return {};
    cv::Rect br = cv::boundingRect(contour);
    int margin = std::max(3, int(std::round(std::max(br.width, br.height) * 0.14f)));
    cv::Mat canvas = cv::Mat::zeros(br.height + 2*margin, br.width + 2*margin, CV_8U);
    std::vector<cv::Point> shifted; shifted.reserve(contour.size());
    for (const auto& p : contour) shifted.emplace_back(p.x - br.x + margin, p.y - br.y + margin);
    cv::drawContours(canvas, std::vector<std::vector<cv::Point>>{shifted}, -1, 255, cv::FILLED);
    std::vector<cv::Point> nz; cv::findNonZero(canvas, nz);
    if (nz.size() < 6) return {};
    cv::RotatedRect rr = cv::minAreaRect(nz);
    if (rr.size.width < 2.0f || rr.size.height < 2.0f) return {};
    cv::Point2f pts[4]; rr.points(pts);
    auto ordered = orderQuad(pts);
    std::vector<cv::Point2f> src{ordered[0],ordered[1],ordered[2],ordered[3]};
    std::vector<cv::Point2f> dst{{4,4},{67,4},{67,67},{4,67}};
    cv::Mat M = cv::getPerspectiveTransform(src,dst);
    cv::Mat warped;
    cv::warpPerspective(canvas,warped,M,{72,72},cv::INTER_NEAREST,cv::BORDER_CONSTANT,cv::Scalar(0));
    cv::threshold(warped,warped,1,255,cv::THRESH_BINARY);
    return warped;
}

HTopologyMetrics HGeometryDetector::topologyOneOrientation(const cv::Mat& mask, HMode mode) const {
    HTopologyMetrics r;
    if (mask.empty()) return r;
    float lt=zoneMean01(mask,.08f,.32f,.08f,.40f), lb=zoneMean01(mask,.08f,.32f,.60f,.92f);
    float rt=zoneMean01(mask,.68f,.92f,.08f,.40f), rb=zoneMean01(mask,.68f,.92f,.60f,.92f);
    float cross=zoneMean01(mask,.20f,.80f,.40f,.60f);
    float vt=1.0f-zoneMean01(mask,.37f,.63f,.08f,.35f);
    float vb=1.0f-zoneMean01(mask,.37f,.63f,.65f,.92f);
    float centre=zoneMean01(mask,.37f,.63f,.42f,.58f);
    float railMin=std::min({lt,lb,rt,rb}); float railAvg=.25f*(lt+lb+rt+rb);

    float rminT,ravgT,crossT,voidT,centreMax;
    if (mode == HMode::Far) { rminT=.12f; ravgT=.20f; crossT=.25f; voidT=.48f; centreMax=.97f; }
    else if (mode==HMode::Track || mode==HMode::Confirm || mode==HMode::Recovery) {
        rminT=.18f; ravgT=.28f; crossT=.34f; voidT=.58f; centreMax=.94f;
    } else { rminT=.22f; ravgT=.33f; crossT=.40f; voidT=.68f; centreMax=.90f; }

    r.valid = railMin>=rminT && railAvg>=ravgT && cross>=crossT && vt>=voidT && vb>=voidT && centre<=centreMax;
    r.score = .22f*std::clamp(railMin/.42f,0.0f,1.0f)
            + .16f*std::clamp(railAvg/.50f,0.0f,1.0f)
            + .28f*std::clamp(cross/.60f,0.0f,1.0f)
            + .17f*std::clamp(vt/.90f,0.0f,1.0f)
            + .17f*std::clamp(vb/.90f,0.0f,1.0f);
    r.score=std::clamp(r.score,0.0f,1.0f); r.railMin=railMin; r.railAvg=railAvg; r.crossbar=cross;
    r.voidTop=vt; r.voidBottom=vb; r.centreFill=centre; return r;
}

HTopologyMetrics HGeometryDetector::topology(const cv::Mat& normalizedMask, HMode mode) const {
    auto a=topologyOneOrientation(normalizedMask,mode); cv::Mat rot; cv::rotate(normalizedMask,rot,cv::ROTATE_90_CLOCKWISE);
    auto b=topologyOneOrientation(rot,mode); return a.score>=b.score?a:b;
}

std::optional<Candidate> HGeometryDetector::findH(
    const cv::Mat& bgr, std::optional<cv::Rect2f> anchorBox,
    std::optional<cv::Point2f> predictedCenter, std::optional<cv::Rect2f> predictedBox,
    HMode mode) const {
    if (bgr.empty()) return std::nullopt;
    const int width=bgr.cols,height=bgr.rows;
    std::optional<cv::Rect2f> ref=predictedBox?predictedBox:anchorBox;
    cv::Rect2f roi=searchBox(bgr,anchorBox,predictedBox,mode);
    cv::Rect ir(roi); if(ir.area()<=0)return std::nullopt;
    cv::Mat crop=bgr(ir); int maxSide=std::max(crop.cols,crop.rows);
    int maxAllowed=(!ref || mode==HMode::Acquire)?cfg_.h_global_max_side:cfg_.h_local_max_side;
    float scale=1.0f; cv::Mat work;
    if(maxSide>maxAllowed){scale=float(maxAllowed)/float(maxSide);cv::resize(crop,work,{},scale,scale,cv::INTER_AREA);}else work=crop;
    const float inv=1.0f/std::max(scale,1e-6f);
    std::optional<float> expected;
    if(predictedBox) expected=rectSpan(*predictedBox)*scale;
    cv::Mat mask=contrastMask(work,expected,!ref.has_value());
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    if(contours.empty())return std::nullopt;
    std::optional<cv::Point2f> anchorCenter;if(anchorBox)anchorCenter=rectCenter(*anchorBox);
    std::vector<Candidate> candidates;

    auto evaluateContour = [&](const std::vector<cv::Point>& c,cv::Point offset,bool nested)
        -> std::optional<Candidate> {
        const float area=float(cv::contourArea(c));
        if(area<(nested?6.0f:10.0f))return std::nullopt;
        const cv::Rect localBr=cv::boundingRect(c);
        const cv::Rect br(localBr.x+offset.x,localBr.y+offset.y,localBr.width,localBr.height);
        const float span=float(std::max(br.width,br.height));
        const float shortv=float(std::min(br.width,br.height));
        if(shortv<(nested?3.0f:4.0f)||span>0.55f*std::max(work.cols,work.rows)||span/std::max(1.0f,shortv)>4.3f)
            return std::nullopt;
        const float fill=area/std::max(1.0f,float(localBr.area()));
        std::vector<cv::Point> hull;cv::convexHull(c,hull);
        const float ha=float(cv::contourArea(hull));
        const float sol=area/std::max(1.0f,ha);
        if(nested){
            if(fill<.06f||fill>.72f||sol>.78f)return std::nullopt;
        }else if(mode==HMode::Acquire){
            if(fill<.09f||fill>.60f||sol>.60f)return std::nullopt;
        }else if(fill<.07f||fill>.70f||sol>.74f)return std::nullopt;

        cv::Mat norm=normalizedComponentMask(c);
        if(norm.empty())return std::nullopt;
        auto tm=topology(norm,mode);
        if(!tm.valid)return std::nullopt; // Aryan positive authority is preserved.

        const float rawW=std::max(1.0f,br.width*inv);
        const float rawH=std::max(1.0f,br.height*inv);
        const float rawSpan=std::max(rawW,rawH);
        const auto core=evaluateCoreLock05(norm,rawSpan);

        // CoreLock_05 resolved three-stroke integrity is a hard NEGATIVE guard.
        if(core.integrity.hardChecked && !core.integrity.pass)return std::nullopt;
        // Below resolved scale, CoreLock_05 FAR signature protects against the
        // C/blob family. Very tiny <8 px candidates retain Aryan's proven path.
        if(!core.integrity.hardChecked && rawSpan>=8.0f &&
           core.signature.kind==helipad_corelock05::FarSignatureKind::Reject)
            return std::nullopt;

        const float spanScore=std::clamp(span/36.0f,0.0f,1.0f);
        const float areaScore=std::clamp(float(std::log1p(area)/std::log1p(900.0)),0.0f,1.0f);
        const float legacyShape=std::clamp(.76f*tm.score+.18f*spanScore+.06f*areaScore,0.0f,1.0f);
        const float coreQ=coreLockQuality(core,rawSpan);
        const float haloQ=nested?1.0f:localHaloQuietScore(mask,br);
        float shape=std::clamp(.83f*legacyShape+.14f*coreQ+.03f*haloQ,0.0f,1.0f);
        if(nested)shape=std::clamp(.99f*shape,0.0f,1.0f);

        const float gx=roi.x+(br.x+.5f*br.width)*inv;
        const float gy=roi.y+(br.y+.5f*br.height)*inv;
        const cv::Point2f center(gx,gy);
        float temporal=0.0f,distance=0.0f;
        if(predictedCenter){
            distance=cv::norm(center-*predictedCenter);
            const float pspan=predictedBox?rectSpan(*predictedBox):rawSpan;
            float radius;
            if(mode==HMode::Recovery)radius=std::max(90.0f,4.5f*pspan);
            else if(mode==HMode::Confirm)radius=std::max(30.0f,1.55f*pspan);
            else radius=std::max(26.0f,1.25f*pspan);
            if(distance>radius)return std::nullopt;
            temporal=std::clamp(1.0f-distance/radius,0.0f,1.0f);
            if(predictedBox){
                const float ratio=std::max(rawSpan/std::max(1.0f,pspan),pspan/std::max(1.0f,rawSpan));
                if(mode!=HMode::Recovery&&ratio>2.15f)return std::nullopt;
            }
        }else if(anchorCenter)distance=cv::norm(center-*anchorCenter);

        float finalScore;
        if(predictedCenter)finalScore=.82f*shape+.18f*temporal;
        else if(!anchorCenter && mode==HMode::Acquire){
            const float nx=(gx-.5f*width)/std::max(1.0f,.45f*width);
            const float ny=(gy-.5f*height)/std::max(1.0f,.45f*height);
            const float centerPrior=std::exp(-(nx*nx+ny*ny));
            const float yr=gy/std::max(1.0f,float(height));
            const float gp=yr<.18f?.35f:(yr<.30f?.65f:1.0f);
            finalScore=.88f*shape+.12f*(centerPrior*gp);
        }else finalScore=shape;
        finalScore=std::clamp(finalScore,0.0f,1.0f);
        if(finalScore<hIdentityThreshold(mode))return std::nullopt;

        const cv::Rect2f raw=clampRect({roi.x+br.x*inv,roi.y+br.y*inv,br.width*inv,br.height*inv},width,height);
        const float pad=std::max(4.0f,.30f*rectSpan(raw));
        const cv::Rect2f box=clampRect({raw.x-pad,raw.y-pad,raw.width+2*pad,raw.height+2*pad},width,height);
        Candidate out;out.center=center;out.box=box;out.rawBox=raw;out.score=finalScore;
        out.shapeScore=shape;out.topology=tm.score;out.span=rawSpan;out.area=area*inv*inv;
        out.distance=distance;out.fill=fill;out.solidity=sol;out.railMin=tm.railMin;
        out.railAvg=tm.railAvg;out.crossbar=tm.crossbar;out.voidTop=tm.voidTop;
        out.voidBottom=tm.voidBottom;out.centreFill=tm.centreFill;out.source=CandidateSource::PlainH;
        return out;
    };

    // Fast path first: unchanged candidate generation + Aryan H topology,
    // strengthened only by CoreLock_05 negative integrity/FAR evidence.
    for(const auto& c:contours){
        if(auto q=evaluateContour(c,{0,0},false))candidates.push_back(*q);
    }

    // V15.54 BOUNDED NESTED-H RESCUE
    // Only when the direct path found no valid H.  This mirrors CoreLock_05's
    // 46% -> 36% -> 58% centered probes and is capped at four support contours.
    // The nested crop cannot create identity by itself: evaluateContour still
    // requires Aryan H topology + CoreLock integrity/signature.
    if(candidates.empty()){
        struct Support{int index;float priority;};
        std::vector<Support> supports;
        for(int i=0;i<int(contours.size());++i){
            const auto& c=contours[i];
            const float area=float(cv::contourArea(c));if(area<18.0f)continue;
            const cv::Rect br=cv::boundingRect(c);
            const float major=std::max(br.width,br.height)*inv;
            const float minor=std::min(br.width,br.height)*inv;
            if(major<kCoreLockStrictMajorPx||minor<6.0f)continue;
            if(float(std::max(br.width,br.height))/std::max(1.0f,float(std::min(br.width,br.height)))>4.5f)continue;
            float priority=float(std::log1p(area));
            if(predictedCenter){
                const cv::Point2f center(roi.x+(br.x+.5f*br.width)*inv,roi.y+(br.y+.5f*br.height)*inv);
                const float d=cv::norm(center-*predictedCenter);
                const float r=std::max(80.0f,3.5f*major);
                priority+=4.0f*std::exp(-d/std::max(1.0f,r));
            }
            supports.push_back({i,priority});
        }
        std::sort(supports.begin(),supports.end(),[](const Support&a,const Support&b){return a.priority>b.priority;});
        if(supports.size()>4)supports.resize(4);
        const auto scales=helipad_corelock05::nestedProbeScales();
        for(const auto& support:supports){
            const cv::Rect br=cv::boundingRect(contours[support.index]);
            for(double sf:scales){
                const int w=std::max(6,int(std::lround(br.width*sf)));
                const int h=std::max(6,int(std::lround(br.height*sf)));
                cv::Rect inner(int(std::lround(br.x+.5*br.width-.5*w)),int(std::lround(br.y+.5*br.height-.5*h)),w,h);
                inner&=cv::Rect(0,0,mask.cols,mask.rows);
                if(inner.width<6||inner.height<6)continue;
                cv::Mat innerMask=mask(inner).clone();
                std::vector<std::vector<cv::Point>> innerContours;
                cv::findContours(innerMask,innerContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                std::sort(innerContours.begin(),innerContours.end(),[](const auto&a,const auto&b){return cv::contourArea(a)>cv::contourArea(b);});
                if(innerContours.size()>6)innerContours.resize(6);
                for(const auto& nc:innerContours){
                    if(auto q=evaluateContour(nc,inner.tl(),true))candidates.push_back(*q);
                }
                if(!candidates.empty())break;
            }
            if(!candidates.empty())break;
        }
    }

    if(candidates.empty())return std::nullopt;
    std::sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){return a.score>b.score;});
    return candidates.front();
}

std::vector<Candidate> HGeometryDetector::farCandidatesOneScale(
    const cv::Mat& bgr,const cv::Rect2f& roi,int maxSide,
    std::optional<cv::Point2f> predictedCenter,std::optional<float> predictedSpan) const {
    cv::Rect ir(clampRect(roi,bgr.cols,bgr.rows));if(ir.area()<=0)return{};
    cv::Mat crop=bgr(ir);float scale=std::min(1.0f,float(maxSide)/float(std::max(crop.cols,crop.rows)));
    cv::Mat work;if(scale<.999f)cv::resize(crop,work,{},scale,scale,cv::INTER_AREA);else work=crop;
    cv::Mat mask=farContrastMask(work);std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    const float inv=1.0f/std::max(scale,1e-6f);std::vector<Candidate> out;
    for(const auto& c:contours){
        const float area=float(cv::contourArea(c));if(area<4)continue;
        const cv::Rect br=cv::boundingRect(c);const float span=float(std::max(br.width,br.height)),shortv=float(std::min(br.width,br.height));
        if(shortv<3||span/std::max(1.0f,shortv)>4.3f||span>.30f*std::max(work.cols,work.rows))continue;
        const float fill=area/std::max(1.0f,float(br.area()));std::vector<cv::Point> hull;cv::convexHull(c,hull);
        const float ha=float(cv::contourArea(hull)),sol=area/std::max(1.0f,ha);if(fill<.07f||fill>.68f||sol>.72f)continue;
        const float rawW=std::max(1.0f,br.width*inv),rawH=std::max(1.0f,br.height*inv),rawSpan=std::max(rawW,rawH);
        if(rawSpan<cfg_.h_far_min_raw_span||rawSpan>cfg_.h_far_max_raw_span)continue;
        cv::Mat norm=normalizedComponentMask(c);if(norm.empty())continue;
        auto tm=topology(norm,HMode::Far);if(!tm.valid)continue;

        // CoreLock_05 FAR signature is specifically designed to reject the
        // C/blob false family before it can accumulate Aryan's existing 3-hit FAR authority.
        const auto core=evaluateCoreLock05(norm,rawSpan);
        if(core.signature.kind==helipad_corelock05::FarSignatureKind::Reject)continue;
        if(core.integrity.hardChecked&&!core.integrity.pass)continue;

        const float gx=roi.x+(br.x+.5f*br.width)*inv,gy=roi.y+(br.y+.5f*br.height)*inv;
        const float dx=(gx-.5f*bgr.cols)/std::max(1.0f,.33f*bgr.cols),dy=(gy-.5f*bgr.rows)/std::max(1.0f,.34f*bgr.rows);
        const float cp=std::exp(-(dx*dx+dy*dy));const float lr=std::log(std::max(rawSpan,1.0f)/18.0f);
        const float sp=std::exp(-.55f*lr*lr);const float conc=std::clamp((.72f-sol)/.55f,0.0f,1.0f);
        float legacy=std::clamp(.80f*tm.score+.11f*cp+.05f*sp+.04f*conc,0.0f,1.0f);
        const cv::Point2f center(gx,gy);float distance=0.0f;
        if(predictedCenter){
            distance=cv::norm(center-*predictedCenter);const float radius=std::max(cfg_.h_far_match_radius,3.0f*predictedSpan.value_or(rawSpan));
            if(distance>radius)continue;const float temporal=std::clamp(1.0f-distance/radius,0.0f,1.0f);
            legacy=std::clamp(.78f*tm.score+.18f*temporal+.04f*sp,0.0f,1.0f);
        }
        const float score=std::clamp(.88f*legacy+.12f*coreLockQuality(core,rawSpan),0.0f,1.0f);
        const cv::Rect2f raw=clampRect({roi.x+br.x*inv,roi.y+br.y*inv,br.width*inv,br.height*inv},bgr.cols,bgr.rows);
        const float pad=std::max(4.0f,.42f*rectSpan(raw));const cv::Rect2f box=clampRect({raw.x-pad,raw.y-pad,raw.width+2*pad,raw.height+2*pad},bgr.cols,bgr.rows);
        Candidate q;q.center=center;q.box=box;q.rawBox=raw;q.score=score;q.shapeScore=score;q.topology=tm.score;q.span=rawSpan;
        q.area=area*inv*inv;q.distance=distance;q.fill=fill;q.solidity=sol;q.railMin=tm.railMin;q.railAvg=tm.railAvg;
        q.crossbar=tm.crossbar;q.voidTop=tm.voidTop;q.voidBottom=tm.voidBottom;q.centreFill=tm.centreFill;q.source=CandidateSource::FarH;
        out.push_back(q);
    }
    std::sort(out.begin(),out.end(),[](const auto&a,const auto&b){return a.score>b.score;});
    if(int(out.size())>cfg_.h_far_candidates_per_scale)out.resize(cfg_.h_far_candidates_per_scale);
    return out;
}

std::optional<Candidate> HGeometryDetector::farNativeVerify(const cv::Mat& bgr,const Candidate& c0) const {
    const float span=c0.span;const int half=int(std::max(28.0f,2.6f*span));
    const int x1=std::max(0,int(std::round(c0.center.x))-half),y1=std::max(0,int(std::round(c0.center.y))-half);
    const int x2=std::min(bgr.cols,int(std::round(c0.center.x))+half),y2=std::min(bgr.rows,int(std::round(c0.center.y))+half);
    if(x2<=x1||y2<=y1)return std::nullopt;
    cv::Mat crop=bgr(cv::Rect(x1,y1,x2-x1,y2-y1));cv::Mat mask=farContrastMask(crop);
    std::vector<std::vector<cv::Point>> contours;cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    std::optional<Candidate> best;float bestScore=-1.0f;
    for(const auto& c:contours){
        const float area=float(cv::contourArea(c));if(area<4)continue;
        const cv::Rect br=cv::boundingRect(c);const float rawSpan=float(std::max(br.width,br.height)),shortv=float(std::min(br.width,br.height));
        if(shortv<3||rawSpan/std::max(1.0f,shortv)>4.3f)continue;
        const cv::Point2f center(x1+br.x+.5f*br.width,y1+br.y+.5f*br.height);const float dist=cv::norm(center-c0.center);
        if(dist>std::max(10.0f,.90f*span))continue;
        const float fill=area/std::max(1.0f,float(br.area()));std::vector<cv::Point> hull;cv::convexHull(c,hull);
        const float ha=float(cv::contourArea(hull)),sol=area/std::max(1.0f,ha);if(fill<.07f||fill>.70f||sol>.74f)continue;
        cv::Mat norm=normalizedComponentMask(c);if(norm.empty())continue;auto tm=topology(norm,HMode::Far);if(!tm.valid)continue;
        const bool strict=tm.railMin>=.16f&&tm.railAvg>=.25f&&tm.crossbar>=.30f&&tm.voidTop>=.58f&&tm.voidBottom>=.58f&&tm.centreFill<=.97f;
        if(!strict)continue;
        const auto core=evaluateCoreLock05(norm,rawSpan);
        if(core.signature.kind==helipad_corelock05::FarSignatureKind::Reject)continue;
        if(core.integrity.hardChecked&&!core.integrity.pass)continue;
        const float proximity=std::clamp(1.0f-dist/std::max(10.0f,.90f*span),0.0f,1.0f);
        const float conc=std::clamp((.72f-sol)/.50f,0.0f,1.0f);
        const float legacy=std::clamp(.82f*tm.score+.10f*conc+.08f*proximity,0.0f,1.0f);
        const float ns=std::clamp(.86f*legacy+.14f*coreLockQuality(core,rawSpan),0.0f,1.0f);
        const cv::Rect2f raw=clampRect({float(x1+br.x),float(y1+br.y),float(br.width),float(br.height)},bgr.cols,bgr.rows);
        const float pad=std::max(4.0f,.42f*rectSpan(raw));const cv::Rect2f box=clampRect({raw.x-pad,raw.y-pad,raw.width+2*pad,raw.height+2*pad},bgr.cols,bgr.rows);
        Candidate q=c0;q.center=center;q.box=box;q.rawBox=raw;q.span=rectSpan(raw);q.area=area;q.topology=tm.score;q.shapeScore=ns;q.score=ns;
        q.fill=fill;q.solidity=sol;q.railMin=tm.railMin;q.railAvg=tm.railAvg;q.crossbar=tm.crossbar;q.voidTop=tm.voidTop;q.voidBottom=tm.voidBottom;q.centreFill=tm.centreFill;
        if(ns>bestScore){best=q;bestScore=ns;}
    }
    return best;
}

std::optional<Candidate> HGeometryDetector::findFar(
    const cv::Mat& bgr,std::optional<cv::Point2f> predictedCenter,
    std::optional<cv::Rect2f> predictedBox,std::optional<int> scaleSide) const {
    if(!cfg_.h_far_enabled||bgr.empty())return std::nullopt;cv::Rect2f roi(cfg_.h_far_x1*bgr.cols,cfg_.h_far_y1*bgr.rows,(cfg_.h_far_x2-cfg_.h_far_x1)*bgr.cols,(cfg_.h_far_y2-cfg_.h_far_y1)*bgr.rows);int side=scaleSide.value_or(cfg_.h_far_scale_sides.front());std::optional<float> pspan;if(predictedBox)pspan=rectSpan(*predictedBox);auto coarse=farCandidatesOneScale(bgr,roi,side,predictedCenter,pspan);if(coarse.empty())return std::nullopt;std::vector<Candidate> verified;int limit=std::min<int>(cfg_.h_far_native_verify_limit,coarse.size());
    for(int i=0;i<limit;++i){auto native=farNativeVerify(bgr,coarse[i]);if(!native)continue;float score=.62f*coarse[i].score+.38f*native->score;if(predictedCenter){float dist=cv::norm(native->center-*predictedCenter);float radius=std::max(cfg_.h_far_match_radius,3.0f*pspan.value_or(native->span));if(dist>radius)continue;float temporal=std::clamp(1.0f-dist/radius,0.0f,1.0f);score=.70f*score+.30f*temporal;}Candidate q=coarse[i];q.center=native->center;q.box=native->box;q.rawBox=native->rawBox;q.span=native->span;q.area=native->area;q.topology=native->topology;q.shapeScore=std::clamp(score,0.0f,1.0f);q.score=q.shapeScore;q.fill=native->fill;q.solidity=native->solidity;q.railMin=native->railMin;q.railAvg=native->railAvg;q.crossbar=native->crossbar;q.voidTop=native->voidTop;q.voidBottom=native->voidBottom;q.centreFill=native->centreFill;q.source=CandidateSource::FarH;verified.push_back(q);}
    if(verified.empty())return std::nullopt;std::sort(verified.begin(),verified.end(),[](const auto&a,const auto&b){return a.score>b.score;});float thr=predictedCenter?cfg_.h_far_confirm_score:cfg_.h_far_acquire_score;if(verified.front().score<thr)return std::nullopt;return verified.front();
}


std::optional<Candidate> HGeometryDetector::findOblique(
    const cv::Mat& bgr,
    std::optional<cv::Point2f> predictedCenter,
    std::optional<cv::Rect2f> predictedBox) const {
    if (!cfg_.h_oblique_enabled || bgr.empty()) return std::nullopt;

    // Global oblique search uses a wider ground corridor than FAR_H.  FAR_H's
    // y2=0.66 is intentionally retained for tiny targets, but it clips a large
    // H as the aircraft approaches.  The oblique path therefore reaches 0.92H.
    cv::Rect2f roi;
    if (predictedBox) {
        roi = expandRect(*predictedBox, 6.0f, bgr.cols, bgr.rows);
    } else {
        roi = cv::Rect2f(
            cfg_.h_oblique_x1 * bgr.cols,
            cfg_.h_oblique_y1 * bgr.rows,
            (cfg_.h_oblique_x2 - cfg_.h_oblique_x1) * bgr.cols,
            (cfg_.h_oblique_y2 - cfg_.h_oblique_y1) * bgr.rows);
    }
    cv::Rect ir(clampRect(roi, bgr.cols, bgr.rows));
    if (ir.area() <= 0) return std::nullopt;

    auto oneScale = [&](int maxSide) {
        std::vector<Candidate> out;
        cv::Mat crop = bgr(ir);
        const float scale = std::min(
            1.0f, float(maxSide) / float(std::max(crop.cols, crop.rows)));
        cv::Mat work;
        if (scale < 0.999f)
            cv::resize(crop, work, {}, scale, scale, cv::INTER_AREA);
        else
            work = crop;

        cv::Mat mask = farContrastMask(work);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        const float inv = 1.0f / std::max(scale, 1e-6f);
        const float predictedSpan = predictedBox ? rectSpan(*predictedBox) : 0.0f;

        // Cheap ranking before perspective normalization.  The oblique path is
        // intentionally for medium/large markers after the FAR_H grace period;
        // FAR_H remains responsible for the tiny 7-90 px case.  Capping this
        // shortlist prevents hundreds of 72x72 affine tests on 4K grass texture.
        struct Rough {
            int index; float area; cv::Rect br; float fill; float solidity;
            float rawSpan; float cheap;
        };
        std::vector<Rough> rough;
        rough.reserve(contours.size());
        for (int ci=0; ci<int(contours.size()); ++ci) {
            const auto& c=contours[ci];
            const float area=float(cv::contourArea(c));
            if(area<4.0f)continue;
            const cv::Rect br=cv::boundingRect(c);
            const float span=float(std::max(br.width,br.height));
            const float shortSide=float(std::min(br.width,br.height));
            if(shortSide<3.0f||span/std::max(1.0f,shortSide)>5.0f)continue;
            if(span>0.35f*std::max(work.cols,work.rows))continue;
            const float rawSpan=span*inv;
            if(rawSpan<cfg_.h_oblique_min_raw_span||rawSpan>cfg_.h_oblique_max_raw_span)continue;
            const float fill=area/std::max(1.0f,float(br.area()));
            std::vector<cv::Point> hull;cv::convexHull(c,hull);
            const float hullArea=float(cv::contourArea(hull));
            const float solidity=area/std::max(1.0f,hullArea);
            if(fill<0.06f||fill>0.72f||solidity>0.75f)continue;
            const float spanRank=std::clamp((rawSpan-20.0f)/150.0f,0.0f,1.0f);
            const float concavity=std::clamp((0.75f-solidity)/0.55f,0.0f,1.0f);
            const float fillRank=std::exp(-5.0f*std::fabs(fill-0.34f));
            rough.push_back({ci,area,br,fill,solidity,rawSpan,
                             0.42f*spanRank+0.38f*concavity+0.20f*fillRank});
        }
        std::sort(rough.begin(),rough.end(),[](const Rough&a,const Rough&b){
            return a.cheap>b.cheap;
        });
        if(rough.size()>72)rough.resize(72);

        for (const Rough& rc : rough) {
            const auto& c = contours[rc.index];
            const float area = rc.area;
            const cv::Rect br = rc.br;
            const float rawSpan = rc.rawSpan;
            const float fill = rc.fill;
            const float solidity = rc.solidity;

            cv::Mat norm = normalizedComponentMask(c);
            if (norm.empty()) continue;

            // Projective ground views are locally close to an affine shear.
            // V10's minAreaRect normalization fixes rotation/scale but not this
            // shear.  Evaluate only nine small affine corrections and retain the
            // strongest exact H topology.  This does not relax the normal/far
            // detector and is only used by the guarded OBLIQUE_H path.
            HTopologyMetrics best = topology(norm, HMode::Far);
            const float baseVoid = std::min(best.voidTop, best.voidBottom);
            // Do not spend affine work on generic texture.  Supplied difficult
            // oblique H samples still have a strong pre-shear crossbar/void
            // signature; the shear is needed mainly to restore the weak rail.
            if (!best.valid &&
                (best.score < 0.55f || baseVoid < 0.42f ||
                 best.crossbar < 0.28f || best.railAvg < 0.18f))
                continue;
            if (!best.valid || best.score < 0.92f) {
                static const float sxVals[] = {-0.50f, 0.0f, 0.50f};
                static const float syVals[] = {-0.30f, 0.0f, 0.30f};
                for (float sx : sxVals) {
                    for (float sy : syVals) {
                        cv::Mat A = (cv::Mat_<double>(2,3) <<
                            1.0, sx, -sx * 0.5 * norm.rows,
                            sy, 1.0, -sy * 0.5 * norm.cols);
                        cv::Mat warped;
                        cv::warpAffine(norm, warped, A, norm.size(),
                                       cv::INTER_NEAREST, cv::BORDER_CONSTANT,
                                       cv::Scalar(0));
                        HTopologyMetrics tm = topology(warped, HMode::Far);
                        if (tm.score > best.score) best = tm;
                    }
                }
            }

            const float voidMin = std::min(best.voidTop, best.voidBottom);
            const bool perspectiveH = best.valid ||
                (best.score >= 0.72f &&
                 voidMin >= 0.55f &&
                 best.railAvg >= 0.28f &&
                 best.crossbar >= 0.38f &&
                 best.centreFill <= 0.95f);
            if (!perspectiveH) continue;

            const float gx = ir.x + (br.x + 0.5f * br.width) * inv;
            const float gy = ir.y + (br.y + 0.5f * br.height) * inv;
            const cv::Point2f center(gx, gy);

            float temporal = 0.0f;
            float distance = 0.0f;
            if (predictedCenter) {
                distance = cv::norm(center - *predictedCenter);
                const float radius = std::max(
                    45.0f, 1.8f * std::max(predictedSpan, rawSpan));
                if (distance > radius) continue;
                temporal = std::clamp(1.0f - distance / radius, 0.0f, 1.0f);
            }

            const float spanTerm = std::clamp((rawSpan - 20.0f) / 110.0f, 0.0f, 1.0f);
            const float concavity = std::clamp((0.72f - solidity) / 0.50f, 0.0f, 1.0f);
            float score =
                0.40f * best.score +
                0.20f * voidMin +
                0.12f * best.railAvg +
                0.10f * best.crossbar +
                0.08f * concavity +
                0.10f * spanTerm;
            if (predictedCenter) score = 0.88f * score + 0.12f * temporal;
            score = std::clamp(score, 0.0f, 1.0f);

            cv::Rect2f raw = clampRect(
                {ir.x + br.x * inv, ir.y + br.y * inv,
                 br.width * inv, br.height * inv}, bgr.cols, bgr.rows);
            const float pad = std::max(4.0f, 0.34f * rectSpan(raw));

            Candidate q;
            q.center = center;
            q.rawBox = raw;
            q.box = clampRect(
                {raw.x - pad, raw.y - pad,
                 raw.width + 2.0f * pad, raw.height + 2.0f * pad},
                bgr.cols, bgr.rows);
            q.score = score;
            q.shapeScore = best.score;
            q.topology = best.score;
            q.span = rawSpan;
            q.area = area * inv * inv;
            q.distance = distance;
            q.fill = fill;
            q.solidity = solidity;
            q.railMin = best.railMin;
            q.railAvg = best.railAvg;
            q.crossbar = best.crossbar;
            q.voidTop = best.voidTop;
            q.voidBottom = best.voidBottom;
            q.centreFill = best.centreFill;
            q.source = CandidateSource::ObliqueH;
            out.push_back(q);
        }

        std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });
        if (out.size() > 80) out.resize(80);
        return out;
    };

    if (cfg_.h_oblique_scale_sides.empty()) return std::nullopt;
    const int s0 = cfg_.h_oblique_scale_sides.front();
    const int s1 = cfg_.h_oblique_scale_sides.size() > 1
        ? cfg_.h_oblique_scale_sides[1] : s0;
    const auto a = oneScale(s0);
    const auto b = oneScale(s1);
    if (a.empty() || b.empty()) return std::nullopt;

    std::optional<Candidate> bestPair;
    float bestPairScore = -1.0f;
    for (const auto& x : a) {
        for (const auto& y : b) {
            const float dist = cv::norm(x.center - y.center);
            const float radius = std::max(30.0f, 0.40f * std::max(x.span, y.span));
            if (dist > radius) continue;
            const float sizeRatio = std::max(
                x.span / std::max(1.0f, y.span),
                y.span / std::max(1.0f, x.span));
            if (sizeRatio > 1.70f) continue;

            const float agree = std::exp(-std::pow(dist / std::max(10.0f, radius), 2.0f));
            const float score = std::clamp(
                0.45f * x.score + 0.45f * y.score + 0.10f * agree,
                0.0f, 1.0f);
            if (score <= bestPairScore) continue;

            Candidate q = (x.score >= y.score) ? x : y;
            q.center = 0.5f * (x.center + y.center);
            q.box = clampRect({
                0.5f * (x.box.x + y.box.x),
                0.5f * (x.box.y + y.box.y),
                0.5f * (x.box.width + y.box.width),
                0.5f * (x.box.height + y.box.height)}, bgr.cols, bgr.rows);
            q.rawBox = clampRect({
                0.5f * (x.rawBox.x + y.rawBox.x),
                0.5f * (x.rawBox.y + y.rawBox.y),
                0.5f * (x.rawBox.width + y.rawBox.width),
                0.5f * (x.rawBox.height + y.rawBox.height)}, bgr.cols, bgr.rows);
            q.span = 0.5f * (x.span + y.span);
            q.score = score;
            q.shapeScore = 0.5f * (x.shapeScore + y.shapeScore);
            q.topology = 0.5f * (x.topology + y.topology);
            q.railMin = 0.5f * (x.railMin + y.railMin);
            q.railAvg = 0.5f * (x.railAvg + y.railAvg);
            q.crossbar = 0.5f * (x.crossbar + y.crossbar);
            q.voidTop = 0.5f * (x.voidTop + y.voidTop);
            q.voidBottom = 0.5f * (x.voidBottom + y.voidBottom);
            q.centreFill = 0.5f * (x.centreFill + y.centreFill);
            q.source = CandidateSource::ObliqueH;
            bestPair = q;
            bestPairScore = score;
        }
    }

    if (!bestPair) return std::nullopt;
    const float threshold = predictedCenter
        ? cfg_.h_oblique_confirm_score : cfg_.h_oblique_acquire_score;
    if (bestPair->score < threshold) return std::nullopt;
    return bestPair;
}

std::optional<Candidate> HGeometryDetector::findColorInvariant(
    const cv::Mat& bgr,std::optional<cv::Rect2f> anchorBox,
    std::optional<cv::Point2f> predictedCenter,std::optional<cv::Rect2f> predictedBox,
    HMode mode) const {
    if(!cfg_.universal_color_enabled||bgr.empty())return std::nullopt;
    cv::Rect2f roi=searchBox(bgr,anchorBox,predictedBox,mode);cv::Rect ir(roi);if(ir.area()<=0)return std::nullopt;
    cv::Mat crop=bgr(ir);int maxAllowed=cfg_.universal_color_max_side;
    if(predictedBox||anchorBox)maxAllowed=std::min(maxAllowed,cfg_.h_local_max_side);
    float scale=std::min(1.0f,float(maxAllowed)/float(std::max(crop.cols,crop.rows)));cv::Mat work;
    if(scale<.999f)cv::resize(crop,work,{},scale,scale,cv::INTER_AREA);else work=crop;
    const float inv=1.0f/std::max(scale,1e-6f);std::optional<float> expected;
    if(predictedBox)expected=rectSpan(*predictedBox)*scale;

    cv::Mat gray;cv::cvtColor(work,gray,cv::COLOR_BGR2GRAY);double sigma=expected?std::clamp(double(*expected)*.16,4.0,24.0):7.0;
    cv::Mat blur;cv::GaussianBlur(gray,blur,{0,0},sigma);cv::Mat g16,b16,diff;gray.convertTo(g16,CV_16S);blur.convertTo(b16,CV_16S);cv::subtract(g16,b16,diff);
    std::vector<float> absvals;absvals.reserve(diff.total()/4+1);
    for(int y=0;y<diff.rows;y+=2){const short* row=diff.ptr<short>(y);for(int x=0;x<diff.cols;x+=2)absvals.push_back(std::fabs(float(row[x])));}
    int gthr=int(std::clamp(std::max(5.0f,percentile(absvals,82.0f)),5.0f,22.0f));cv::Mat pos,neg;
    cv::compare(diff,cv::Scalar(gthr),pos,cv::CMP_GE);cv::compare(diff,cv::Scalar(-gthr),neg,cv::CMP_LE);

    cv::Mat lab,labf,labBlur;cv::cvtColor(work,lab,cv::COLOR_BGR2Lab);lab.convertTo(labf,CV_32F);cv::GaussianBlur(labf,labBlur,{0,0},sigma);
    cv::Mat delta(work.rows,work.cols,CV_32F);std::vector<float> dv;dv.reserve(delta.total()/4+1);
    for(int y=0;y<work.rows;++y){const cv::Vec3f*a=labf.ptr<cv::Vec3f>(y);const cv::Vec3f*b=labBlur.ptr<cv::Vec3f>(y);float*o=delta.ptr<float>(y);for(int x=0;x<work.cols;++x){cv::Vec3f q=a[x]-b[x];o[x]=std::sqrt(q.dot(q));if((x%2==0)&&(y%2==0))dv.push_back(o[x]);}}
    float lthr=std::clamp(std::max(5.5f,percentile(dv,84.0f)),5.5f,26.0f);cv::Mat chroma;cv::compare(delta,cv::Scalar(lthr),chroma,cv::CMP_GE);

    std::vector<cv::Mat> masks{pos,neg,chroma};std::vector<Candidate> candidates;
    for(auto mask:masks){
        cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,cv::Mat::ones(3,3,CV_8U));
        if(std::min(mask.rows,mask.cols)>=80)cv::morphologyEx(mask,mask,cv::MORPH_OPEN,cv::Mat::ones(2,2,CV_8U));
        std::vector<std::vector<cv::Point>> contours;cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
        for(const auto& c:contours){
            const float area=float(cv::contourArea(c));if(area<8)continue;const cv::Rect br=cv::boundingRect(c);
            const float span=float(std::max(br.width,br.height)),shortv=float(std::min(br.width,br.height));
            if(shortv<3||span/std::max(1.0f,shortv)>4.6f||span>.48f*std::max(mask.cols,mask.rows))continue;
            const float fill=area/std::max(1.0f,float(br.area()));std::vector<cv::Point> hull;cv::convexHull(c,hull);
            const float ha=float(cv::contourArea(hull)),sol=area/std::max(1.0f,ha);if(fill<.06f||fill>.76f||sol>.80f)continue;
            cv::Mat norm=normalizedComponentMask(c);if(norm.empty())continue;
            const HMode topoMode=(mode==HMode::Acquire?HMode::Acquire:HMode::Recovery);auto tm=topology(norm,topoMode);if(!tm.valid)continue;
            const float gx=roi.x+(br.x+.5f*br.width)*inv,gy=roi.y+(br.y+.5f*br.height)*inv;const cv::Point2f center(gx,gy);
            const float rawSpan=std::max(br.width,br.height)*inv;
            const auto core=evaluateCoreLock05(norm,rawSpan);
            if(core.integrity.hardChecked&&!core.integrity.pass)continue;
            if(!core.integrity.hardChecked&&rawSpan>=8.0f&&core.signature.kind==helipad_corelock05::FarSignatureKind::Reject)continue;

            float temporal=0.0f,distance=0.0f;
            if(predictedCenter){
                distance=cv::norm(center-*predictedCenter);const float pspan=predictedBox?rectSpan(*predictedBox):rawSpan;
                const float radius=mode==HMode::Recovery?std::max(100.0f,4.8f*pspan):(mode==HMode::Confirm?std::max(38.0f,1.8f*pspan):std::max(32.0f,1.5f*pspan));
                if(distance>radius)continue;temporal=std::clamp(1.0f-distance/radius,0.0f,1.0f);
            }
            const float spanScore=std::clamp(rawSpan/42.0f,0.0f,1.0f),conc=std::clamp((.82f-sol)/.62f,0.0f,1.0f);
            const float legacyShape=std::clamp(.82f*tm.score+.10f*spanScore+.08f*conc,0.0f,1.0f);
            const float shape=std::clamp(.87f*legacyShape+.13f*coreLockQuality(core,rawSpan),0.0f,1.0f);
            float finalScore;
            if(predictedCenter)finalScore=.84f*shape+.16f*temporal;
            else{
                const float nx=(gx-.5f*bgr.cols)/std::max(1.0f,.46f*bgr.cols),ny=(gy-.52f*bgr.rows)/std::max(1.0f,.46f*bgr.rows);
                const float cp=std::exp(-(nx*nx+ny*ny)),yr=gy/std::max(1.0f,float(bgr.rows)),gp=std::clamp((yr-.10f)/.35f,.25f,1.0f);
                finalScore=.91f*shape+.06f*cp+.03f*gp;
            }
            const float threshold=(mode==HMode::Acquire?cfg_.universal_color_h_min:cfg_.universal_local_h_min);if(finalScore<threshold)continue;
            const cv::Rect2f raw=clampRect({roi.x+br.x*inv,roi.y+br.y*inv,br.width*inv,br.height*inv},bgr.cols,bgr.rows);
            const float pad=std::max(4.0f,.34f*rectSpan(raw));Candidate q;q.center=center;q.rawBox=raw;
            q.box=clampRect({raw.x-pad,raw.y-pad,raw.width+2*pad,raw.height+2*pad},bgr.cols,bgr.rows);
            q.score=std::clamp(finalScore,0.0f,1.0f);q.shapeScore=shape;q.topology=tm.score;q.span=rawSpan;q.area=area*inv*inv;
            q.distance=distance;q.fill=fill;q.solidity=sol;q.railMin=tm.railMin;q.railAvg=tm.railAvg;q.crossbar=tm.crossbar;
            q.voidTop=tm.voidTop;q.voidBottom=tm.voidBottom;q.centreFill=tm.centreFill;q.source=CandidateSource::GenericH;candidates.push_back(q);
        }
    }
    if(candidates.empty())return std::nullopt;
    std::sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){return a.score>b.score;});return candidates.front();
}

RingHelper::RingHelper(VisionConfig cfg,const HGeometryDetector& hDetector):cfg_(std::move(cfg)),hDetector_(hDetector){}

std::optional<Candidate> RingHelper::findRingH(
    const cv::Mat& bgr,
    std::optional<cv::Rect2f> anchorBox,
    std::optional<cv::Point2f> predictedCenter,
    std::optional<cv::Rect2f> predictedBox,
    HMode mode) const
{

    // ========================================================
    // V15.35X DEEP SPLIT PROFILE
    // PROFILING ONLY - detector behavior is unchanged.
    // ========================================================

    using V35XClock =
        std::chrono::steady_clock;

    const auto v35xFn0 =
        V35XClock::now();

    double v35xCanonicalMs = 0.0;
    double v35xSerialYawMs = 0.0;

    // ========================================================
    // V15.15 ANGLE-INVARIANT HELIPAD DETECTOR
    //
    // IMPORTANT:
    //
    //   circle/ellipse = proposal/support
    //   H              = identity
    //   H center       = guidance center
    //
    // FAR_H / PLAIN_H / OBLIQUE_H are NOT modified here.
    // ========================================================

    if(!cfg_.universal_ring_enabled || bgr.empty())
        return std::nullopt;


    // ========================================================
    // ROI
    // ========================================================

    cv::Rect2f roi;

    if(predictedBox){

        roi = expandRect(
            *predictedBox,
            mode==HMode::Recovery ? 8.0f : 5.0f,
            bgr.cols,
            bgr.rows);

    }else if(anchorBox){

        roi = expandRect(
            *anchorBox,
            mode==HMode::Recovery ? 8.0f : 5.0f,
            bgr.cols,
            bgr.rows);

    }else{

        roi = {
            0.0f,
            0.0f,
            float(bgr.cols),
            float(bgr.rows)
        };
    }


    cv::Rect ir(
        clampRect(
            roi,
            bgr.cols,
            bgr.rows));

    if(ir.area()<=0)
        return std::nullopt;

    cv::Mat crop = bgr(ir);


    // ========================================================
    // SEARCH RESOLUTION
    //
    // <= 2K:
    // retain more helipad detail.
    //
    // 4K:
    // retain existing cost/performance behaviour.
    // ========================================================

    int maxAllowed;

    if(anchorBox || predictedBox){

        maxAllowed =
            std::min(
                cfg_.universal_ring_max_side,
                900);

    }else{

        const int sourceSide =
            std::max(
                crop.cols,
                crop.rows);

        maxAllowed =
            sourceSide<=2200
                ? std::max(
                      cfg_.universal_ring_max_side,
                      1600)
                : cfg_.universal_ring_max_side;
    }


    const float scale =
        std::min(
            1.0f,
            float(maxAllowed) /
            float(
                std::max(
                    crop.cols,
                    crop.rows)));

    const float inv =
        1.0f /
        std::max(
            scale,
            1e-6f);


    cv::Mat work;

    if(scale<0.999f){

        cv::resize(
            crop,
            work,
            {},
            scale,
            scale,
            cv::INTER_AREA);

    }else{

        work = crop;
    }


    // ========================================================
    // RING EDGE PROPOSAL
    // ========================================================

    cv::Mat gray;

    cv::cvtColor(
        work,
        gray,
        cv::COLOR_BGR2GRAY);


    cv::Mat gb;

    cv::GaussianBlur(
        gray,
        gb,
        {5,5},
        0);


    std::vector<uchar> medv;

    medv.reserve(
        gb.total()/4 + 1);

    for(int y=0; y<gb.rows; y+=2){

        const uchar* r =
            gb.ptr<uchar>(y);

        for(int x=0; x<gb.cols; x+=2)
            medv.push_back(r[x]);
    }


    std::vector<float> mf;

    mf.reserve(
        medv.size());

    for(auto x : medv)
        mf.push_back(x);


    const float med =
        median(mf);

    const int low =
        int(
            std::clamp(
                0.42f*med,
                12.0f,
                95.0f));

    const int high =
        int(
            std::clamp(
                1.20f*med,
                float(low+20),
                220.0f));


    cv::Mat edges;

    cv::Canny(
        gb,
        edges,
        low,
        high,
        3,
        true);


    // --------------------------------------------------------
    // Fine edge channel.
    //
    // Preserve thin/small helipad circles during acquisition.
    // --------------------------------------------------------

    if(mode==HMode::Acquire &&
       !anchorBox &&
       !predictedBox){

        cv::Mat fineBlur;

        cv::GaussianBlur(
            gray,
            fineBlur,
            {3,3},
            0);

        const int fineLow =
            std::max(
                6,
                int(0.65f*low));

        const int fineHigh =
            std::max(
                fineLow+12,
                int(0.85f*high));

        cv::Mat fineEdges;

        cv::Canny(
            fineBlur,
            fineEdges,
            fineLow,
            fineHigh,
            3,
            true);

        cv::bitwise_or(
            edges,
            fineEdges,
            edges);
    }


    std::vector<
        std::vector<cv::Point>
    > contours;

    cv::findContours(
        edges,
        contours,
        cv::RETR_LIST,
        cv::CHAIN_APPROX_NONE);


    struct E
    {
        cv::RotatedRect e;

        float areaProxy = 0.0f;
        float residual  = 0.0f;
        float pair      = 0.0f;
    };


    std::vector<E> es;


    // ========================================================
    // ELLIPSE PROPOSALS
    // ========================================================

    for(const auto& c : contours){

        const size_t minPoints =
            mode==HMode::Acquire
                ? 12
                : 20;

        const double minArc =
            mode==HMode::Acquire
                ? 16.0
                : 28.0;

        if(c.size()<minPoints)
            continue;

        if(cv::arcLength(c,false)<minArc)
            continue;


        cv::RotatedRect e;

        try{

            e = cv::fitEllipse(c);

        }catch(...){

            continue;
        }


        const float mn =
            std::min(
                e.size.width,
                e.size.height);

        const float mx =
            std::max(
                e.size.width,
                e.size.height);


        if(mn <
           cfg_.universal_ring_min_axis*
           scale)
            continue;


        if(mx >
           0.62f*
           std::max(
               work.cols,
               work.rows))
            continue;


        const float aspect =
            mx /
            std::max(
                1.0f,
                mn);

        if(aspect >
           cfg_.universal_ring_max_aspect)
            continue;


        const float th =
            e.angle *
            float(CV_PI/180.0);

        const float ca =
            std::cos(th);

        const float sa =
            std::sin(th);


        std::vector<float> residuals;

        residuals.reserve(
            c.size());


        for(const auto& q : c){

            const float x =
                q.x -
                e.center.x;

            const float y =
                q.y -
                e.center.y;

            const float xr =
                x*ca +
                y*sa;

            const float yr =
                -x*sa +
                y*ca;

            const float rr =
                std::sqrt(
                    std::pow(
                        xr /
                        std::max(
                            1.0f,
                            0.5f*
                            e.size.width),
                        2.0f)
                    +
                    std::pow(
                        yr /
                        std::max(
                            1.0f,
                            0.5f*
                            e.size.height),
                        2.0f));

            residuals.push_back(
                std::fabs(
                    rr-1.0f));
        }


        const float res =
            median(
                residuals);


        if(res >
           cfg_.universal_ring_fit_residual)
            continue;


        es.push_back(
            {
                e,
                e.size.width*
                e.size.height,
                res,
                0.0f
            });
    }


    if(es.empty())
        return std::nullopt;


    // ========================================================
    // CONCENTRIC EDGE PAIRING
    //
    // A painted ring usually provides inner + outer ellipse.
    //
    // Critical V15.14/15.15 correction:
    // a nearly circular ellipse has no reliable orientation.
    // ========================================================

    
    // ========================================================
    // V15.35Y EXACT DEEP-RING SPATIAL PAIR
    //
    // PERFORMANCE ONLY.
    //
    // ORIGINAL pair tests, thresholds and update math below
    // remain untouched.
    //
    // A pair is skipped only when its centres are farther
    // apart than the GLOBAL maximum possible original
    // centre-distance threshold.
    //
    // Candidate j indices are sorted into original ascending
    // order before the original pair body executes.
    // ========================================================

    constexpr float V35Y_CELL = 32.0f;

    const int v35yGridW =
        std::max(
            1,
            int(
                std::ceil(
                    float(work.cols) /
                    V35Y_CELL)));

    const int v35yGridH =
        std::max(
            1,
            int(
                std::ceil(
                    float(work.rows) /
                    V35Y_CELL)));

    std::vector<
        std::vector<size_t>
    > v35yBins(
        size_t(v35yGridW) *
        size_t(v35yGridH));

    float v35yMaxCentreTolerance =
        6.0f;


    auto v35yCellX =
        [&](float x)
        {
            return std::clamp(
                int(
                    std::floor(
                        x /
                        V35Y_CELL)),
                0,
                v35yGridW - 1);
        };


    auto v35yCellY =
        [&](float y)
        {
            return std::clamp(
                int(
                    std::floor(
                        y /
                        V35Y_CELL)),
                0,
                v35yGridH - 1);
        };


    for(size_t k=0;
        k<es.size();
        ++k)
    {
        const float mn =
            std::min(
                es[k].e.size.width,
                es[k].e.size.height);

        // Original pair threshold is:
        //
        // max(6.0f, 0.20f * outerMinor)
        //
        // Using the maximum over every ellipse makes this
        // spatial pre-filter conservative: it cannot remove
        // any pair that the original code could accept.
        v35yMaxCentreTolerance =
            std::max(
                v35yMaxCentreTolerance,
                0.20f * mn);

        const int cx =
            v35yCellX(
                es[k].e.center.x);

        const int cy =
            v35yCellY(
                es[k].e.center.y);

        v35yBins[
            size_t(cy) *
            size_t(v35yGridW) +
            size_t(cx)
        ].push_back(k);
    }


    const int v35yRadiusCells =
        std::max(
            1,
            int(
                std::ceil(
                    v35yMaxCentreTolerance /
                    V35Y_CELL)));


    size_t v35yGridPairChecks = 0;


    for(size_t i=0;
        i<es.size();
        ++i)
    {
        const int cx =
            v35yCellX(
                es[i].e.center.x);

        const int cy =
            v35yCellY(
                es[i].e.center.y);


        std::vector<size_t>
            v35yCandidates;


        const int y0 =
            std::max(
                0,
                cy - v35yRadiusCells);

        const int y1 =
            std::min(
                v35yGridH - 1,
                cy + v35yRadiusCells);

        const int x0 =
            std::max(
                0,
                cx - v35yRadiusCells);

        const int x1 =
            std::min(
                v35yGridW - 1,
                cx + v35yRadiusCells);


        for(int gy=y0;
            gy<=y1;
            ++gy)
        {
            for(int gx=x0;
                gx<=x1;
                ++gx)
            {
                const auto& bin =
                    v35yBins[
                        size_t(gy) *
                        size_t(v35yGridW) +
                        size_t(gx)];


                for(const size_t j : bin)
                {
                    if(j > i)
                    {
                        v35yCandidates.push_back(
                            j);
                    }
                }
            }
        }


        // Restore EXACT original j processing order.
        std::sort(
            v35yCandidates.begin(),
            v35yCandidates.end());


        for(const size_t j :
            v35yCandidates)
        {
            ++v35yGridPairChecks;



            auto& a = es[i];
            auto& b = es[j];

            E* outer =
                a.areaProxy>=b.areaProxy
                    ? &a
                    : &b;

            E* inner =
                a.areaProxy>=b.areaProxy
                    ? &b
                    : &a;


            const float omin =
                std::min(
                    outer->e.size.width,
                    outer->e.size.height);


            const float d =
                cv::norm(
                    outer->e.center -
                    inner->e.center);


            if(d >
               std::max(
                   5.0f,
                   cfg_.universal_ring_pair_center*
                   omin))
                continue;


            const float ra =
                inner->e.size.width /
                std::max(
                    1.0f,
                    outer->e.size.width);

            const float rb =
                inner->e.size.height /
                std::max(
                    1.0f,
                    outer->e.size.height);


            if(ra<0.40f ||
               ra>0.97f ||
               rb<0.40f ||
               rb>0.97f)
                continue;


            const float oa =
                std::max(
                    outer->e.size.width,
                    outer->e.size.height)
                /
                std::max(
                    1.0f,
                    std::min(
                        outer->e.size.width,
                        outer->e.size.height));

            const float ia =
                std::max(
                    inner->e.size.width,
                    inner->e.size.height)
                /
                std::max(
                    1.0f,
                    std::min(
                        inner->e.size.width,
                        inner->e.size.height));


            if(std::fabs(oa-ia)>0.80f)
                continue;


            const bool bothNearlyCircular =
                oa<=1.30f &&
                ia<=1.30f;


            // Circle orientation is mathematically ambiguous.
            // Only enforce orientation for real ellipses.
            if(!bothNearlyCircular &&
               angleDelta180(
                   outer->e.angle,
                   inner->e.angle)>35.0f)
                continue;


            const float align =
                std::exp(
                    -std::pow(
                        d /
                        std::max(
                            3.0f,
                            0.13f*omin),
                        2.0f));


            const float sm =
                std::exp(
                    -3.0f*
                    std::fabs(
                        ra-rb));


            const float ps =
                align*sm;


            outer->pair =
                std::max(
                    outer->pair,
                    ps);

            inner->pair =
                std::max(
                    inner->pair,
                    0.92f*ps);
        
        }
    }


    if(kRingProfiling &&
       mode==HMode::Acquire &&
       !anchorBox &&
       !predictedBox)
    {
        const size_t v35yAllPairs =
            es.size() > 1
                ? (
                    es.size() *
                    (es.size()-1)
                  ) / 2
                : 0;

        std::cerr
            << "[YPAIR]"
            << " es=" << es.size()
            << " all_pairs="
            << v35yAllPairs
            << " grid_pairs="
            << v35yGridPairChecks
            << " max_tol="
            << v35yMaxCentreTolerance
            << " radius_cells="
            << v35yRadiusCells
            << std::endl;
    }




    // Sort strong concentric rings first.
    std::sort(
        es.begin(),
        es.end(),
        [](const E& a,const E& b){

            if(std::fabs(a.pair-b.pair)>0.05f)
                return a.pair>b.pair;

            if(std::fabs(a.residual-b.residual)>0.01f)
                return a.residual<b.residual;

            return a.areaProxy>b.areaProxy;
        });


    // Keep cost bounded.
    if(es.size()>48)
        es.resize(48);


    std::optional<Candidate> best;

    // V15.35V profiler accumulators for THIS findRingH call.
    // Must live outside the per-ellipse loop because VPROF
    // is printed after all ellipse verification is complete.
    double v35vAccumMs = 0.0;
    size_t v35vAccumBatches = 0;

    // ========================================================
    // V15.35V2 ACQUIRE-ONLY PARALLEL
    //
    // Six-way yaw parallelism only belongs to the expensive
    // full-frame DEEP_RING Acquire path.
    //
    // Local Confirm / Recovery keeps the original serial
    // ColorInvariant -> findH fallback behavior.
    // ========================================================

    const bool v35vParallel =
        mode==HMode::Acquire &&
        !anchorBox &&
        !predictedBox;



    // ========================================================
    // VERIFY H INSIDE EACH ELLIPSE
    // ========================================================

    
    const auto v35xVerify0 =
        V35XClock::now();

    // ========================================================
    // V15.35Z GLOBAL SIX-WORKER VERIFY
    //
    // PERFORMANCE ONLY.
    //
    // All existing:
    //   * 48 ellipses
    //   * 6 yaw hypotheses
    //   * ColorInvariant -> H fallback
    //   * result replay/scoring order
    //
    // are preserved.
    //
    // Difference:
    //   V2 = 6 new async jobs + barrier PER ellipse
    //   Z  = 6 workers process ALL ellipse/yaw jobs
    // ========================================================

    constexpr size_t V35Z_YAWS = 6;
    constexpr int V35Z_CS = 192;

    static const float v35zYawAngles[V35Z_YAWS] =
    {
         0.0f,
        15.0f,
       -15.0f,
        30.0f,
       -30.0f,
        45.0f
    };


    std::vector<cv::Mat>
        v35zCanonical;

    std::vector<
        std::array<
            std::optional<Candidate>,
            V35Z_YAWS
        >
    > v35zResults;


    double v35zCanonicalPrepMs = 0.0;
    double v35zPoolMs = 0.0;


    if(v35vParallel)
    {
        // ----------------------------------------------------
        // Build the exact same canonical image that the
        // original per-ellipse loop builds.
        // ----------------------------------------------------

        const auto v35zCanon0 =
            std::chrono::steady_clock::now();

        v35zCanonical.resize(
            es.size());

        v35zResults.resize(
            es.size());


        for(size_t zi=0;
            zi<es.size();
            ++zi)
        {
            const auto& ze =
                es[zi];

            const float theta =
                ze.e.angle *
                float(CV_PI/180.0);

            const float ca =
                std::cos(theta);

            const float sa =
                std::sin(theta);

            const cv::Point2f u(
                ca,
                sa);

            const cv::Point2f vAxis(
                -sa,
                ca);


            cv::Point2f srcTri[3] =
            {
                ze.e.center,

                ze.e.center +
                    u *
                    (0.5f *
                     ze.e.size.width),

                ze.e.center +
                    vAxis *
                    (0.5f *
                     ze.e.size.height)
            };


            cv::Point2f dstTri[3] =
            {
                {96.0f,96.0f},
                {174.0f,96.0f},
                {96.0f,174.0f}
            };


            const cv::Mat A =
                cv::getAffineTransform(
                    srcTri,
                    dstTri);


            cv::warpAffine(
                work,
                v35zCanonical[zi],
                A,
                {V35Z_CS,V35Z_CS},
                cv::INTER_LINEAR,
                cv::BORDER_REPLICATE);
        }


        const auto v35zCanon1 =
            std::chrono::steady_clock::now();

        v35zCanonicalPrepMs =
            std::chrono::duration<
                double,
                std::milli
            >(
                v35zCanon1 -
                v35zCanon0
            ).count();


        // Tell X profiler where the canonical cost moved.
        v35xCanonicalMs +=
            v35zCanonicalPrepMs;


        // ----------------------------------------------------
        // Six workers process ALL 288 jobs.
        //
        // Dynamic atomic scheduling avoids waiting for the
        // slowest yaw of every individual ellipse.
        // ----------------------------------------------------

        const size_t v35zJobCount =
            es.size() *
            V35Z_YAWS;

        std::atomic<size_t>
            v35zNextJob{0};


        std::array<
            std::future<void>,
            V35Z_YAWS
        > v35zWorkers;


        const auto v35zPool0 =
            std::chrono::steady_clock::now();


        for(size_t worker=0;
            worker<V35Z_YAWS;
            ++worker)
        {
            v35zWorkers[worker] =
                std::async(
                    std::launch::async,
                    [&,worker]()
                    {
                        (void)worker;

                        while(true)
                        {
                            const size_t job =
                                v35zNextJob.fetch_add(
                                    1,
                                    std::memory_order_relaxed);

                            if(job >=
                               v35zJobCount)
                            {
                                break;
                            }


                            const size_t ei =
                                job /
                                V35Z_YAWS;

                            const size_t yi =
                                job %
                                V35Z_YAWS;


                            const float yaw =
                                v35zYawAngles[yi];


                            const cv::Mat&
                                canonical =
                                    v35zCanonical[ei];


                            cv::Mat rotated;


                            if(std::fabs(yaw)<0.01f)
                            {
                                rotated =
                                    canonical;
                            }
                            else
                            {
                                const cv::Mat R =
                                    cv::getRotationMatrix2D(
                                        cv::Point2f(
                                            96.0f,
                                            96.0f),
                                        yaw,
                                        1.0);

                                cv::warpAffine(
                                    canonical,
                                    rotated,
                                    R,
                                    canonical.size(),
                                    cv::INTER_LINEAR,
                                    cv::BORDER_REPLICATE);
                            }


                            const cv::Rect innerRect(
                                30,
                                30,
                                132,
                                132);


                            const cv::Mat inner =
                                rotated(
                                    innerRect);


                            // EXACT original detector order.
                            auto h =
                                hDetector_.
                                    findColorInvariant(
                                        inner,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        HMode::Recovery);


                            if(!h)
                            {
                                h =
                                    hDetector_.findH(
                                        inner,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        HMode::Recovery);
                            }


                            v35zResults[ei][yi] =
                                std::move(h);
                        }
                    });
        }


        for(auto& worker :
            v35zWorkers)
        {
            worker.get();
        }


        const auto v35zPool1 =
            std::chrono::steady_clock::now();


        v35zPoolMs =
            std::chrono::duration<
                double,
                std::milli
            >(
                v35zPool1 -
                v35zPool0
            ).count();


        // Existing X/V profilers expect verification work
        // in these accumulators.
        v35vAccumMs =
            v35zPoolMs;

        v35vAccumBatches =
            1;


        if(kRingProfiling){
            std::cerr
                << "[ZPROF]"
                << " es=" << es.size()
                << " yaw=" << V35Z_YAWS
                << " jobs=" << v35zJobCount
                << " workers=" << V35Z_YAWS
                << " canonical_ms="
                << v35zCanonicalPrepMs
                << " pool_ms="
                << v35zPoolMs
                << std::endl;
        }
    }


    size_t v35zEllipseIndex = 0;



for(auto& e : es){

        const size_t v35zThisEllipse =
            v35zEllipseIndex++;



        const auto v35xCanon0 =
            V35XClock::now();


        const int CS = 192;


        const float theta =
            e.e.angle *
            float(CV_PI/180.0);

        const float ca =
            std::cos(theta);

        const float sa =
            std::sin(theta);


        const cv::Point2f u(
            ca,
            sa);

        const cv::Point2f vAxis(
            -sa,
            ca);


        cv::Point2f srcTri[3] =
        {
            e.e.center,

            e.e.center +
            u*
            (0.5f*e.e.size.width),

            e.e.center +
            vAxis*
            (0.5f*e.e.size.height)
        };


        cv::Point2f dstTri[3] =
        {
            {96.0f,96.0f},
            {174.0f,96.0f},
            {96.0f,174.0f}
        };


        cv::Mat A =
            cv::getAffineTransform(
                srcTri,
                dstTri);


        cv::Mat invA;

        cv::invertAffineTransform(
            A,
            invA);


        cv::Mat canonical;

        
        if(v35vParallel)
        {
            canonical =
                v35zCanonical[
                    v35zThisEllipse];
        }
        else
        {
            cv::warpAffine(
                work,
                canonical,
                A,
                {CS,CS},
                cv::INTER_LINEAR,
                cv::BORDER_REPLICATE);
        }


        v35xCanonicalMs +=
            std::chrono::duration<
                double,
                std::milli
            >(
                V35XClock::now() -
                v35xCanon0
            ).count();



        // ====================================================
        // V15.15 ROTATION-INVARIANT H
        //
        // H topology internally already handles 0/90.
        //
        // These extra canonical rotations therefore cover
        // essentially every in-plane H yaw:
        //
        //    0
        //   ±15
        //   ±30
        //   45
        //
        // Together with internal 90-degree symmetry this covers
        // 0...90 degrees without twelve expensive scans.
        // ====================================================

        static const float yawAngles[] =
        {
             0.0f,
            15.0f,
           -15.0f,
            30.0f,
           -30.0f,
            45.0f
        };


        
        // ====================================================
        // V15.35V PARALLEL YAW VERIFY
        //
        // All six yaw hypotheses are preserved.
        //
        // Within EACH yaw the ORIGINAL detector order remains:
        //
        //   findColorInvariant()
        //        |
        //        +-- if absent --> findH()
        //
        // Results are later consumed in the ORIGINAL yaw order,
        // preserving deterministic candidate/scoring order.
        //
        // Recommended runtime:
        //
        //   OPENCV_FOR_THREADS_NUM=1
        //
        // so these six outer jobs own CPU parallelism instead of
        // producing nested OpenCV/TBB oversubscription.
        // ====================================================

        using V35VClock =
            std::chrono::steady_clock;

                constexpr size_t V35V_YAWS = 6;

        std::array<
            std::future<std::optional<Candidate>>,
            V35V_YAWS
        > v35vFuture;

        std::array<
            std::optional<Candidate>,
            V35V_YAWS
        > v35vResult;

        /* V15.35V2: yaw index declared immediately before consume loop */


        
        if(v35vParallel)
        {
            // V15.35Z:
            // Results were already calculated by the six
            // global workers. Copy them into the existing V2
            // replay container so all downstream geometry /
            // scoring code remains untouched.

            for(size_t yi=0;
                yi<V35Z_YAWS;
                ++yi)
            {
                v35vResult[yi] =
                    std::move(
                        v35zResults[
                            v35zThisEllipse
                        ][yi]);
            }
        }


        size_t v35vYawIndex = 0;


        const auto v35xSerial0 =
            V35XClock::now();

for(const float yaw : yawAngles){

            cv::Mat rotated;
            cv::Mat R;
            cv::Mat invR;


            if(std::fabs(yaw)<0.01f){

                rotated =
                    canonical;

                R =
                    cv::Mat::eye(
                        2,
                        3,
                        CV_64F);

                invR =
                    R.clone();

            }else{

                R =
                    cv::getRotationMatrix2D(
                        cv::Point2f(
                            96.0f,
                            96.0f),
                        yaw,
                        1.0);

                if(!v35vParallel)
                {
                    cv::warpAffine(
                        canonical,
                        rotated,
                        R,
                        canonical.size(),
                        cv::INTER_LINEAR,
                        cv::BORDER_REPLICATE);
                }

                cv::invertAffineTransform(
                    R,
                    invR);
            }


            const cv::Rect innerRect(
                30,
                30,
                132,
                132);


            cv::Mat inner;

            if(!v35vParallel)
            {
                inner =
                    rotated(
                        innerRect);
            }


            // -----------------------------------------------
            // H identity is mandatory.
            // -----------------------------------------------

            std::optional<Candidate> h;

            if(v35vParallel)
            {
                h =
                    std::move(
                        v35vResult[
                            v35vYawIndex++
                        ]);
            }
            else
            {
                // Original LOCAL Confirm / Recovery behavior.

                h =
                    hDetector_.findColorInvariant(
                        inner,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        HMode::Recovery);

                if(!h)
                {
                    h =
                        hDetector_.findH(
                            inner,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            HMode::Recovery);
                }
            }


            if(!h)
                continue;


            // H coordinates:
            // inner ROI -> rotated canonical.
            h->center +=
                cv::Point2f(
                    float(innerRect.x),
                    float(innerRect.y));

            h->rawBox.x +=
                float(innerRect.x);

            h->rawBox.y +=
                float(innerRect.y);


            // -----------------------------------------------
            // ROTATED canonical -> normal canonical
            // -----------------------------------------------

            std::vector<cv::Point2f> rotatedPoints =
            {
                h->center,

                {
                    h->rawBox.x,
                    h->rawBox.y
                },

                {
                    h->rawBox.x+
                    h->rawBox.width,
                    h->rawBox.y
                },

                {
                    h->rawBox.x+
                    h->rawBox.width,
                    h->rawBox.y+
                    h->rawBox.height
                },

                {
                    h->rawBox.x,
                    h->rawBox.y+
                    h->rawBox.height
                }
            };


            std::vector<cv::Point2f> canonicalPoints;

            cv::transform(
                rotatedPoints,
                canonicalPoints,
                invR);


            if(canonicalPoints.size()!=5)
                continue;


            // -----------------------------------------------
            // Canonical -> ring work image.
            // -----------------------------------------------

            std::vector<cv::Point2f> workPoints;

            cv::transform(
                canonicalPoints,
                workPoints,
                invA);


            if(workPoints.size()!=5)
                continue;


            // -----------------------------------------------
            // Work image -> original frame.
            // -----------------------------------------------

            for(auto& q : workPoints){

                q.x =
                    float(ir.x) +
                    q.x*inv;

                q.y =
                    float(ir.y) +
                    q.y*inv;
            }


            const cv::Point2f center =
                workPoints[0];


            std::vector<cv::Point2f> corners =
            {
                workPoints[1],
                workPoints[2],
                workPoints[3],
                workPoints[4]
            };


            const cv::Rect2f raw =
                rectFromCorners(
                    corners,
                    bgr.cols,
                    bgr.rows);


            const float hSpan =
                rectSpan(
                    raw);


            const float ringMinor =
                std::min(
                    e.e.size.width,
                    e.e.size.height)*
                inv;

            const float ringMajor =
                std::max(
                    e.e.size.width,
                    e.e.size.height)*
                inv;


            const cv::Point2f ringCenter(
                float(ir.x) +
                e.e.center.x*inv,

                float(ir.y) +
                e.e.center.y*inv);


            const float off =
                cv::norm(
                    center -
                    ringCenter);


            // H must actually be geometrically inside ring.
            const float minH =
                predictedCenter
                    ? 5.0f
                    : 6.0f;


            if(hSpan <
               std::max(
                   minH,
                   0.08f*ringMinor))
                continue;


            if(hSpan >
               1.10f*ringMajor)
                continue;


            if(off >
               std::max(
                   12.0f,
                   0.58f*ringMinor))
                continue;


            const float hscore =
                h->score;


            const float pair =
                e.pair;


            const float ellipseAspect =
                ringMajor /
                std::max(
                    1.0f,
                    ringMinor);


            const float fit =
                std::clamp(
                    1.0f -
                    e.residual /
                    std::max(
                        0.01f,
                        cfg_.universal_ring_fit_residual),
                    0.0f,
                    1.0f);


            const float centerAlign =
                std::exp(
                    -std::pow(
                        off /
                        std::max(
                            8.0f,
                            0.35f*ringMinor),
                        2.0f));


            // =================================================
            // SINGLE-RING SUPPORT
            //
            // This is the part your failed V15.15 patch was
            // trying to add.
            //
            // It is now inside the complete function so there
            // is no brittle strictSingle regex.
            //
            // A single ellipse is NOT enough:
            //
            //   strong H
            //   + clean ellipse
            //   + H near ellipse centre
            //
            // are all mandatory.
            // =================================================

            const bool startupSingle =
                pair<0.20f &&
                !predictedCenter &&
                mode==HMode::Acquire &&
                hscore>=0.78f &&
                e.residual<=0.12f &&
                centerAlign>=0.45f &&
                ellipseAspect<=
                    std::min(
                        3.2f,
                        cfg_.universal_ring_max_aspect);


            const bool trackedSingle =
                pair<0.20f &&
                predictedCenter &&
                hscore>=
                    cfg_.universal_ring_single_h_min &&
                e.residual<=0.11f &&
                centerAlign>=0.35f;


            const bool strictSingle =
                startupSingle ||
                trackedSingle;


            if(pair>=0.20f){

                if(hscore <
                   cfg_.universal_ring_h_min)
                    continue;

            }else if(!strictSingle){

                continue;
            }


            // =================================================
            // TEMPORAL CHECK
            // =================================================

            float temporal = 0.0f;


            if(predictedCenter){

                const float pspan =
                    predictedBox
                        ? std::max(
                              ringMajor,
                              rectSpan(
                                  *predictedBox))
                        : ringMajor;


                const float radius =
                    mode==HMode::Recovery
                        ? std::max(
                              110.0f,
                              4.5f*pspan)
                        : std::max(
                              48.0f,
                              2.3f*pspan);


                const float d =
                    cv::norm(
                        center -
                        *predictedCenter);


                if(d>radius)
                    continue;


                temporal =
                    std::clamp(
                        1.0f -
                        d/radius,
                        0.0f,
                        1.0f);
            }


            const float support =
                pair>=0.20f
                    ? pair
                    : 0.55f*fit;


            // H identity dominates score.
            const float score =
                std::clamp(
                    0.68f*hscore
                    +
                    0.14f*support
                    +
                    0.08f*fit
                    +
                    0.06f*centerAlign
                    +
                    0.04f*temporal,
                    0.0f,
                    1.0f);


            const float pad =
                std::max(
                    5.0f,
                    0.40f*hSpan);


            Candidate q =
                *h;


            // H is ALWAYS the target.
            q.center =
                center;

            q.rawBox =
                raw;

            q.box =
                clampRect(
                    {
                        raw.x-pad,
                        raw.y-pad,
                        raw.width+2.0f*pad,
                        raw.height+2.0f*pad
                    },
                    bgr.cols,
                    bgr.rows);

            q.span =
                hSpan;

            q.score =
                score;

            q.shapeScore =
                hscore;

            q.source =
                CandidateSource::RingH;


            // Ring = supporting evidence only.
            q.hasRing =
                true;

            q.ring =
                cv::RotatedRect(
                    ringCenter,
                    {
                        e.e.size.width*inv,
                        e.e.size.height*inv
                    },
                    e.e.angle);

            const float rh =
                0.60f*
                ringMajor;

            q.ringBox =
                clampRect(
                    {
                        ringCenter.x-rh,
                        ringCenter.y-rh,
                        2.0f*rh,
                        2.0f*rh
                    },
                    bgr.cols,
                    bgr.rows);

            q.ringCenter =
                ringCenter;

            q.ringPairScore =
                pair;

            q.ringFitScore =
                fit;


            if(!best ||
               q.score>best->score){

                best =
                    q;
            }


            // Excellent verified H+double-ring.
            // No need to test every remaining yaw for this
            // ellipse.
            if(pair>=0.55f &&
               hscore>=0.90f &&
               centerAlign>=0.70f){

                break;
            }
        }

        v35xSerialYawMs +=
            std::chrono::duration<
                double,
                std::milli
            >(
                V35XClock::now() -
                v35xSerial0
            ).count();

    }

    const auto v35xVerify1 =
        V35XClock::now();

    if(v35vParallel)
    {
        const double v35xPreMs =
            std::chrono::duration<
                double,
                std::milli
            >(
                v35xVerify0 -
                v35xFn0
            ).count();

        const double v35xVerifyMs =
            std::chrono::duration<
                double,
                std::milli
            >(
                v35xVerify1 -
                v35xVerify0
            ).count();

        const double v35xMeasuredMs =
            v35xCanonicalMs +
            v35vAccumMs +
            v35xSerialYawMs;

        const double v35xOtherMs =
            std::max(
                0.0,
                v35xVerifyMs -
                v35xMeasuredMs);

        const double v35xTotalMs =
            std::chrono::duration<
                double,
                std::milli
            >(
                v35xVerify1 -
                v35xFn0
            ).count();

        if(kRingProfiling){
            std::cerr
                << "[XSPLIT]"
                << " es=" << es.size()
                << " pre_ms=" << v35xPreMs
                << " canonical_ms=" << v35xCanonicalMs
                << " batch_ms=" << v35vAccumMs
                << " serial_yaw_ms=" << v35xSerialYawMs
                << " other_ms=" << v35xOtherMs
                << " verify_ms=" << v35xVerifyMs
                << " total_ms=" << v35xTotalMs
                << std::endl;
        }
    }



    
    if(kRingProfiling &&
       mode==HMode::Acquire &&
       !anchorBox &&
       !predictedBox)
    {
        std::cerr
            << "[VPROF]"
            << " es=" << es.size()
            << " yaw=" << (6*es.size())
            << " batches=" << v35vAccumBatches
            << " batch_total_ms=" << v35vAccumMs
            << " avg_batch_ms="
            << (
                v35vAccumBatches
                    ? v35vAccumMs /
                      double(v35vAccumBatches)
                    : 0.0
               )
            << " best=" << int(bool(best))
            << std::endl;

    }

return best;
}


bool OpticalTracker::seed(const cv::Mat& gray,const cv::Rect2f& box){clear();if(gray.empty())return false;cv::Rect2f b=clampRect(box,gray.cols,gray.rows);int mx=std::max(1,int(b.width*.06f)),my=std::max(1,int(b.height*.06f));cv::Rect r(int(b.x)+mx,int(b.y)+my,std::max(1,int(b.width)-2*mx),std::max(1,int(b.height)-2*my));r&=cv::Rect(0,0,gray.cols,gray.rows);if(r.area()<=0)return false;cv::Mat roi=gray(r);std::vector<cv::Point2f> local;cv::goodFeaturesToTrack(roi,local,100,.006,3,cv::noArray(),5,false,.04);for(auto&p:local)points_.push_back(p+cv::Point2f(float(r.x),float(r.y)));return int(points_.size())>=3;}

void OpticalTracker::clear(){points_.clear();}

bool OpticalTracker::update(const cv::Mat& previousGray,const cv::Mat& gray,
                            const cv::Rect2f& targetBox,const cv::Point2f& velocity,
                            cv::Point2f& delta,float& quality){
    quality=0.0f;delta={0,0};
    if(previousGray.empty()||gray.empty()||points_.size()<3)return false;

    // V14 performance change: PyrLK no longer builds a 4-level pyramid for the
    // complete 4K frame.  Use the same native-resolution patch in previous and
    // current images.  Points remain in full-image coordinates, so guidance
    // precision and the original V10 motion semantics are unchanged.
    const cv::Point2f c=rectCenter(targetBox);
    const float span=rectSpan(targetBox);
    const float speed=cv::norm(velocity);
    const float half=std::clamp(std::max(96.0f,2.8f*span+4.0f*speed+45.0f),96.0f,900.0f);
    cv::Rect2f rf=clampRect({c.x-half,c.y-half,2.0f*half,2.0f*half},gray.cols,gray.rows);
    cv::Rect r(rf);
    r &= cv::Rect(0,0,gray.cols,gray.rows);
    if(r.area()<=0)return false;

    std::vector<cv::Point2f> localOld;
    localOld.reserve(points_.size());
    for(size_t i=0;i<points_.size();++i){
        const auto&p=points_[i];
        if(p.x>=r.x&&p.x<r.x+r.width&&p.y>=r.y&&p.y<r.y+r.height){
            localOld.push_back(p-cv::Point2f(float(r.x),float(r.y)));
        }
    }
    if(localOld.size()<3)return false;

    const cv::Mat prevPatch=previousGray(r);
    const cv::Mat curPatch=gray(r);
    std::vector<cv::Point2f> localNext;
    std::vector<uchar> status;
    std::vector<float> errors;
    cv::calcOpticalFlowPyrLK(prevPatch,curPatch,localOld,localNext,status,errors,
        {51,51},4,cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT,12,.03));
    if(localNext.empty()||status.empty())return false;

    std::vector<cv::Point2f> oldGood,newGood;
    for(size_t i=0;i<localOld.size()&&i<localNext.size()&&i<status.size();++i){
        if(!status[i])continue;
        oldGood.push_back(localOld[i]+cv::Point2f(float(r.x),float(r.y)));
        newGood.push_back(localNext[i]+cv::Point2f(float(r.x),float(r.y)));
    }
    if(newGood.size()<3)return false;

    std::vector<cv::Point2f> disp;
    std::vector<float> mags;
    for(size_t i=0;i<newGood.size();++i){auto d=newGood[i]-oldGood[i];disp.push_back(d);mags.push_back(cv::norm(d));}
    const float mm=median(mags);
    std::vector<cv::Point2f> keptNew,keptDisp;
    for(size_t i=0;i<disp.size();++i){
        if(mags[i]<std::max(3.0f,mm*2.5f)){keptNew.push_back(newGood[i]);keptDisp.push_back(disp[i]);}
    }
    if(keptNew.size()<3)return false;

    std::vector<float> dx,dy;
    for(const auto&d:keptDisp){dx.push_back(d.x);dy.push_back(d.y);}
    const float mdx=median(dx),mdy=median(dy);
    std::vector<float> residual;
    for(const auto&d:keptDisp)residual.push_back(cv::norm(d-cv::Point2f(mdx,mdy)));
    const float consistency=median(residual);
    const float pq=float(keptNew.size())/float(std::max<size_t>(1,points_.size()));
    quality=std::clamp(pq*std::exp(-consistency/8.0f),0.0f,1.0f);
    delta={mdx,mdy};points_=std::move(keptNew);return true;
}

Guidance calculateGuidance(const cv::Point2f& t,int width,int height,const VisionConfig& cfg){Guidance g;float cx=.5f*width,cy=.5f*height;g.errorX=t.x-cx;g.errorY=cy-t.y;g.normalizedX=g.errorX/std::max(1.0f,cx);g.normalizedY=g.errorY/std::max(1.0f,cy);g.roll=std::clamp(g.normalizedX*cfg.roll_gain,-cfg.max_roll,cfg.max_roll);g.pitch=std::clamp(g.normalizedY*cfg.pitch_gain,-cfg.max_pitch,cfg.max_pitch);return g;}

} // namespace helipad
