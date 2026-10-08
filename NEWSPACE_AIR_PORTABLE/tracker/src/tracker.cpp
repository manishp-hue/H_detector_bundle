#include "tracker.hpp"
#include "corelock05_primary.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <sstream>
#include <utility>

namespace helipad {
namespace {

cv::Rect2f blendRect(const cv::Rect2f& a, const cv::Rect2f& b, float wa, float wb,
                     int width, int height) {
    return clampRect({wa*a.x+wb*b.x, wa*a.y+wb*b.y,
                      wa*a.width+wb*b.width, wa*a.height+wb*b.height}, width, height);
}

cv::Rect2f unionRect(const cv::Rect2f& a, const cv::Rect2f& b, int width, int height) {
    float x1=std::min(a.x,b.x),y1=std::min(a.y,b.y);
    float x2=std::max(a.x+a.width,b.x+b.width),y2=std::max(a.y+a.height,b.y+b.height);
    return clampRect({x1,y1,x2-x1,y2-y1},width,height);
}

HAcquisitionMethod methodFromSource(CandidateSource source) {
    switch(source) {
        case CandidateSource::FarH: return HAcquisitionMethod::Far;
        case CandidateSource::ObliqueH: return HAcquisitionMethod::Oblique;
        case CandidateSource::RingH: return HAcquisitionMethod::RingRoi;
        case CandidateSource::AiVerifiedH: return HAcquisitionMethod::Ai;
        // V15.55: GenericH is the classical color-invariant geometry rescue.
        // It must never be reported as AI when GDINO/SAM2 are disabled.
        case CandidateSource::GenericH:
        case CandidateSource::PlainH: return HAcquisitionMethod::Plain;
        default: return HAcquisitionMethod::None;
    }
}

CandidateSource canonicalHSource(HAcquisitionMethod method) {
    switch(method) {
        case HAcquisitionMethod::Far: return CandidateSource::FarH;
        case HAcquisitionMethod::Oblique: return CandidateSource::ObliqueH;
        case HAcquisitionMethod::Ai: return CandidateSource::AiVerifiedH;
        case HAcquisitionMethod::RingRoi:
        case HAcquisitionMethod::Plain: return CandidateSource::PlainH;
        default: return CandidateSource::PlainH;
    }
}

constexpr double AI_ACTIVE_SUBMIT_MIN_MS = 20.0;
constexpr double AI_LOCKED_SUBMIT_MIN_MS = 220.0;
constexpr float AI_EVIDENCE_MAX_AGE_MS = 700.0f;
constexpr int AI_SAM_MAX_PROPOSALS = 3;

const char* hProofModeName(const Candidate& c) {
    const float span=std::max(c.span,rectSpan(c.box));
    if(span<=0.0f) return "NONE";
    if(span<12.0f) return "TINY";
    if(span<=30.0f) return "MID";
    return "LARGE";
}

const char* targetClassName(TargetClass c) {
    return c==TargetClass::RingH ? "RING_H" : "PLAIN_H";
}

} // namespace

HelipadTracker::HelipadTracker(VisionConfig cfg)
    : cfg_(std::move(cfg)),
      coreLock05_(std::make_unique<CoreLock05Primary>()),
      hDetector_(cfg_),
      identityDetector_(cfg_),
      ringDetector_(cfg_),
      aiProofDetector_(cfg_),
      ringHelper_(cfg_,ringDetector_) {}

HelipadTracker::~HelipadTracker(){
    stopAiWorker();
}

bool HelipadTracker::setAiEngines(const std::string& dinoPlan,const std::string& samPlan){
    if(aiWorker_.joinable()) return aiDinoReady_ && aiSamReady_;
    if(dinoPlan.empty() || samPlan.empty()) return false;

    const bool loaded=ai_.load(dinoPlan,samPlan);
    aiDinoReady_=loaded && ai_.available();
    aiSamReady_=ai_.samAvailable();
    if(!aiDinoReady_ || !aiSamReady_) return false;

    aiStop_.store(false,std::memory_order_release);
    aiWorker_=std::thread(&HelipadTracker::aiWorkerLoop,this);
    return true;
}

void HelipadTracker::setAiEnabled(bool enabled){
    aiEnabled_.store(enabled,std::memory_order_release);
    if(!enabled){
        std::lock_guard<std::mutex> lock(aiResultMutex_);
        latestAiEvidence_.reset();
        lastDinoScore_=0.0f;
        lastSamScore_=0.0f;
        lastAiResultAgeMs_=-1.0f;
    }
}

bool HelipadTracker::aiEnabled() const{
    return aiEnabled_.load(std::memory_order_acquire);
}

void HelipadTracker::stopAiWorker(){
    aiStop_.store(true,std::memory_order_release);
    aiRequests_.stop();
    if(aiWorker_.joinable()) aiWorker_.join();
}

void HelipadTracker::reset(){
    if(coreLock05_) coreLock05_->reset();
    state_=TrackState::Searching;target_={};prevGray_.release();velocity_={0,0};frame_=0;
    searchStartFrame_=0;acquisitionStartFrame_=0;confirmStartFrame_=-100000;
    lastGlobalHFrame_=lastFarHFrame_=lastHVerifyFrame_=lastRingHFrame_=lastObliqueHFrame_=lastDinoFrame_=-100000;
    farScaleCursor_=0;lostFrames_=identityMisses_=confirmHits_=confirmMisses_=globalConfirmHits_=0;
    farConfirmHits_=farConfirmMisses_=obliqueConfirmHits_=obliqueConfirmMisses_=confirmationFailures_=0;currentHScore_=0;pendingTakeover_.reset();takeoverHits_=0;
    pendingRingTakeover_.reset();ringTakeoverHits_=0;
    hasRingSupport_=false;ringSupportMisses_=0;flow_.clear();
    confirmStartTime_={};lastStrongObliqueProofTime_={};
    lastObliqueFastVerifyTime_={};
    lastConfirmCompetitionTime_={};lastConfirmRingAssistTime_={};
    haveStrongObliqueCandidate_=false;fastObliqueHits_=0;
    currentConfirmAuthority_=0.0f;currentConfirmContrast_=0.0f;
    confirmCompetitionUseFar_=true;
    acquisitionMethod_=HAcquisitionMethod::None;
    targetClass_=TargetClass::PlainH;
    lastHSource_=CandidateSource::PlainH;
    hConfirmTracklet_.reset();
    hConfirmMissesUnified_=0;
    hConfirmStrongContextHits_=0;
    lastHConfirmObservationTime_={};
    ringClassHits_=0;
    lastRingClassObservationTime_={};
    lastRingClassCandidate_={};
    haveRingClassCandidate_=false;
    resetFastRingProof();
    invalidateIdentityAuthority();
}

void HelipadTracker::clearTarget(){
    target_={};flow_.clear();lostFrames_=0;identityMisses_=0;confirmHits_=0;confirmMisses_=0;
    globalConfirmHits_=0;confirmStartFrame_=-100000;currentHScore_=0;farConfirmHits_=0;farConfirmMisses_=0;obliqueConfirmHits_=0;obliqueConfirmMisses_=0;
    pendingTakeover_.reset();takeoverHits_=0;
    pendingRingTakeover_.reset();ringTakeoverHits_=0;
    hasRingSupport_=false;ringSupportMisses_=0;velocity_={0,0};
    confirmStartTime_={};lastStrongObliqueProofTime_={};
    lastObliqueFastVerifyTime_={};
    lastConfirmCompetitionTime_={};lastConfirmRingAssistTime_={};
    haveStrongObliqueCandidate_=false;fastObliqueHits_=0;
    currentConfirmAuthority_=0.0f;currentConfirmContrast_=0.0f;
    confirmCompetitionUseFar_=true;
    acquisitionMethod_=HAcquisitionMethod::None;
    targetClass_=TargetClass::PlainH;
    lastHSource_=CandidateSource::PlainH;
    hConfirmTracklet_.reset();
    hConfirmMissesUnified_=0;
    hConfirmStrongContextHits_=0;
    lastHConfirmObservationTime_={};
    ringClassHits_=0;
    lastRingClassObservationTime_={};
    lastRingClassCandidate_={};
    haveRingClassCandidate_=false;
    resetFastRingProof();
    invalidateIdentityAuthority();
}

void HelipadTracker::rememberRingSupport(const Candidate& c){
    if(!c.hasRing && c.source!=CandidateSource::RingH)return;
    ringSupportBox_=c.ringBox.area()>0?c.ringBox:c.box;
    ringSupportCenter_=c.ringCenter.x!=0||c.ringCenter.y!=0?c.ringCenter:c.center;
    hasRingSupport_=true;ringSupportMisses_=0;
}

void HelipadTracker::seedCandidate(const Candidate& c,const cv::Mat& gray,bool globalSeed,CandidateSource source){
    invalidateIdentityAuthority();

    const auto now=Clock::now();
    acquisitionMethod_=methodFromSource(source);
    targetClass_=TargetClass::PlainH;
    lastHSource_=canonicalHSource(acquisitionMethod_);

    target_=c;
    // CandidateSource is detector-origin/UI metadata only in V15.42.
    target_.source=lastHSource_;
    state_=TrackState::Confirming;
    flow_.seed(gray,target_.box);
    lostFrames_=0;
    identityMisses_=0;

    hConfirmTracklet_.reset();
    HObservation seed;
    seed.x=c.center.x;
    seed.y=c.center.y;
    seed.span=std::max(1.0f,c.span>0.0f?c.span:rectSpan(c.box));
    seed.score=c.score;
    seed.topology=c.topology;
    seed.timeMs=std::chrono::duration<double,std::milli>(now.time_since_epoch()).count();
    seed.method=acquisitionMethod_;
    hConfirmTracklet_.observe(seed);

    confirmHits_=hConfirmTracklet_.hits();
    confirmMisses_=0;
    hConfirmMissesUnified_=0;
    globalConfirmHits_=globalSeed?1:0;
    confirmStartFrame_=frame_;
    currentHScore_=c.score;
    currentConfirmContrast_=robustCandidateContrast(gray,c);
    hConfirmStrongContextHits_=currentConfirmContrast_>=55.0f?1:0;
    lastHConfirmObservationTime_=now;

    farConfirmHits_=source==CandidateSource::FarH?1:0;
    farConfirmMisses_=0;
    obliqueConfirmHits_=source==CandidateSource::ObliqueH?1:0;
    obliqueConfirmMisses_=0;

    confirmStartTime_=now;
    lastConfirmCompetitionTime_=now;
    lastConfirmRingAssistTime_=now;
    confirmCompetitionUseFar_=true;
    resetFastObliqueProof();
    resetFastRingProof();

    ringClassHits_=0;
    lastRingClassObservationTime_={};
    lastRingClassCandidate_={};
    haveRingClassCandidate_=false;

    if(c.hasRing || source==CandidateSource::RingH){
        rememberRingSupport(c);
        ringClassHits_=1;
        lastRingClassCandidate_=c;
        lastRingClassObservationTime_=now;
        haveRingClassCandidate_=true;
        // Keep the RingH classification provisional until a second associated
        // ring observation arrives. H confirmation proceeds independently.
        target_.hasRing=false;
    }else{
        hasRingSupport_=false;
        ringSupportMisses_=0;
        target_.hasRing=false;
    }
}

bool HelipadTracker::correctionAllowed(const cv::Point2f& candidateCenter,float factor) const {
    if(target_.source==CandidateSource::None)return true;float speed=cv::norm(velocity_);float allowed=std::max({20.0f,1.75f*rectSpan(target_.box),3.0f*speed+12.0f})*factor;return cv::norm(candidateCenter-target_.center)<=allowed;
}

void HelipadTracker::aiWorkerLoop(){
    Clock::time_point previousFinish{};

    while(!aiStop_.load(std::memory_order_acquire)){
        AiRequest request;
        if(!aiRequests_.waitTake(request)) break;
        if(aiStop_.load(std::memory_order_acquire)) break;

        AiEvidence out;
        out.generation=request.generation;
        out.serial=request.serial;
        out.recovery=request.recovery;
        out.capturedAt=request.capturedAt;

        try{
            auto proposals=ai_.detect(request.frame,cfg_.ai_top_k);
            std::sort(
                proposals.begin(),
                proposals.end(),
                [](const AiProposal& a,const AiProposal& b){
                    return a.score>b.score;
                });

            const int retained=std::min<int>(
                AI_SAM_MAX_PROPOSALS,
                static_cast<int>(proposals.size()));

            auto toGlobal=[&](cv::Rect2f r){
                r.x+=request.offset.x;
                r.y+=request.offset.y;
                return clampRect(r,request.sourceSize.width,request.sourceSize.height);
            };

            bool promptCovered=false;
            for(int i=0;i<retained;++i){
                const AiProposal& proposal=proposals[static_cast<size_t>(i)];
                float samScore=0.0f;
                const cv::Rect2f refined=
                    ai_.refineWithSam(request.frame,proposal.box,&samScore);

                AiEvidenceItem item;
                item.dinoBox=toGlobal(proposal.box);
                item.samBox=toGlobal(refined);
                item.dinoScore=proposal.score;
                item.samScore=samScore;
                item.fromClassicalPrompt=false;
                out.items.push_back(item);

                if(request.samPrompt){
                    const cv::Rect2f a=proposal.box;
                    const cv::Rect2f b=*request.samPrompt;
                    const float x1=std::max(a.x,b.x);
                    const float y1=std::max(a.y,b.y);
                    const float x2=std::min(a.x+a.width,b.x+b.width);
                    const float y2=std::min(a.y+a.height,b.y+b.height);
                    const float inter=std::max(0.0f,x2-x1)*std::max(0.0f,y2-y1);
                    const float uni=std::max(1.0f,a.area()+b.area()-inter);
                    if(inter/uni>=0.20f) promptCovered=true;
                }
            }

            // A classical FAR/Plain/Oblique observation can wake SAM2 even when
            // GDINO did not return an overlapping proposal. This keeps SAM2
            // central without making it synchronous in the real-time lane.
            if(request.samPrompt && !promptCovered){
                float samScore=0.0f;
                const cv::Rect2f refined=
                    ai_.refineWithSam(request.frame,*request.samPrompt,&samScore);
                AiEvidenceItem item;
                item.dinoBox=toGlobal(*request.samPrompt);
                item.samBox=toGlobal(refined);
                item.dinoScore=0.0f;
                item.samScore=samScore;
                item.fromClassicalPrompt=true;
                out.items.push_back(item);
            }
        }catch(...){
            // Keep the worker alive. Startup already validated both engines; an
            // individual inference failure becomes an empty evidence batch.
        }

        out.finishedAt=Clock::now();
        if(previousFinish.time_since_epoch().count()!=0){
            const double dt=std::chrono::duration<double>(
                out.finishedAt-previousFinish).count();
            if(dt>1.0e-6){
                const float instant=static_cast<float>(1.0/dt);
                const float old=aiFps_.load(std::memory_order_relaxed);
                aiFps_.store(
                    old>0.0f ? 0.75f*old+0.25f*instant : instant,
                    std::memory_order_relaxed);
            }
        }
        previousFinish=out.finishedAt;

        {
            std::lock_guard<std::mutex> lock(aiResultMutex_);
            latestAiEvidence_=std::move(out);
        }
        }
}

void HelipadTracker::submitAiRequest(const cv::Mat& bgr,Clock::time_point now){
    if(!aiEnabled_.load(std::memory_order_acquire) || !aiDinoReady_ || !aiSamReady_ || !aiWorker_.joinable() || bgr.empty())
        return;

    const bool healthyLocked=
        state_==TrackState::Locked &&
        haveIdentityProof_ &&
        identityAgeMs(now)<=0.70f*cfg_.identity_soft_lease_ms;
    const double minPeriodMs=
        healthyLocked ? AI_LOCKED_SUBMIT_MIN_MS : AI_ACTIVE_SUBMIT_MIN_MS;

    if(lastAiSubmitTime_.time_since_epoch().count()!=0){
        const double elapsed=std::chrono::duration<double,std::milli>(
            now-lastAiSubmitTime_).count();
        if(elapsed<minPeriodMs) return;
    }

    AiRequest request;
    request.generation=targetGeneration_;
    request.serial=++aiSerial_;
    request.recovery=state_==TrackState::Suspect;
    request.capturedAt=now;
    request.sourceSize=bgr.size();

    if(target_.source!=CandidateSource::None && state_!=TrackState::Searching){
        const float factor=request.recovery ? 6.0f :
            (state_==TrackState::Confirming ? 5.0f : 4.0f);
        const cv::Rect2f roi=expandRect(
            target_.box,factor,bgr.cols,bgr.rows);
        cv::Rect ir(roi);
        ir &= cv::Rect(0,0,bgr.cols,bgr.rows);
        if(ir.area()>0){
            request.frame=bgr(ir).clone();
            request.offset={float(ir.x),float(ir.y)};
            cv::Rect2f prompt=target_.box;
            prompt.x-=request.offset.x;
            prompt.y-=request.offset.y;
            request.samPrompt=clampRect(
                prompt,request.frame.cols,request.frame.rows);
        }
    }

    if(request.frame.empty()){
        request.frame=bgr.clone();
        request.offset={0,0};
        request.samPrompt.reset();
    }

    if(aiRequests_.submit(std::move(request)))
        lastAiSubmitTime_=now;
}

std::optional<Candidate> HelipadTracker::pollAiEvidence(
    const cv::Mat& bgr,Clock::time_point now){
    // V15.55 SEMANTIC GUARD: no loaded GDINO+SAM2 pair means no AI-assisted
    // evidence can exist. This also prevents any stale queued result from ever
    // being surfaced after AI is disabled.
    if(!aiEnabled_.load(std::memory_order_acquire) || !aiDinoReady_ || !aiSamReady_ || !aiWorker_.joinable()){
        std::lock_guard<std::mutex> lock(aiResultMutex_);
        latestAiEvidence_.reset();
        lastDinoScore_=0.0f;
        lastSamScore_=0.0f;
        return std::nullopt;
    }

    std::optional<AiEvidence> evidence;
    {
        std::lock_guard<std::mutex> lock(aiResultMutex_);
        if(latestAiEvidence_){
            evidence=std::move(latestAiEvidence_);
            latestAiEvidence_.reset();
        }
    }
    if(!evidence) return std::nullopt;

    lastAiEvidenceCaptureTime_=evidence->capturedAt;
    lastAiResultAgeMs_=static_cast<float>(
        std::chrono::duration<double,std::milli>(
            now-evidence->capturedAt).count());
    lastDinoScore_=0.0f;
    lastSamScore_=0.0f;
    for(const auto& item:evidence->items){
        lastDinoScore_=std::max(lastDinoScore_,item.dinoScore);
        lastSamScore_=std::max(lastSamScore_,item.samScore);
    }

    if(evidence->generation!=targetGeneration_) return std::nullopt;
    if(lastAiResultAgeMs_>AI_EVIDENCE_MAX_AGE_MS) return std::nullopt;
    if(bgr.empty()) return std::nullopt;

    // V15.43 CURRENT-FRAME AI H-PROOF
    // AI boxes/masks are immutable evidence from their capture frame. They are
    // never flow-translated into the current frame. Instead they nominate a
    // bounded ROI in which a fresh current-frame H detector must independently
    // prove identity before the result can enter the unified H tracklet.
    std::optional<Candidate> best;
    float bestCombined=-1.0f;
    for(const auto& item:evidence->items){
        cv::Rect2f context=unionRect(
            item.dinoBox,item.samBox,bgr.cols,bgr.rows);
        const float ageExpansion=lastAiResultAgeMs_>350.0f ? 2.25f : 1.55f;
        context=expandRect(context,ageExpansion,bgr.cols,bgr.rows);

        const cv::Point2f predicted=rectCenter(context);
        std::optional<Candidate> h=aiProofDetector_.findH(
            bgr,context,predicted,context,
            evidence->recovery ? HMode::Recovery : HMode::Confirm);
        if(!h && cfg_.universal_color_enabled){
            h=aiProofDetector_.findColorInvariant(
                bgr,context,predicted,context,
                evidence->recovery ? HMode::Recovery : HMode::Confirm);
        }
        if(!h) continue;

        // V15.43 TOP-EDGE PERMISSIVE GUARD: AI PROOF
        if(state_==TrackState::Searching ||
           state_==TrackState::Confirming)
        {
            const float topGuardY =
                std::max(
                    64.0f,
                    0.18f * float(bgr.rows));

            const float proofSpan =
                std::max(
                    h->span,
                    rectSpan(h->box));

            if(proofSpan > 30.0f &&
               h->center.y < topGuardY)
            {
                continue;
            }
        }

        h->source=CandidateSource::AiVerifiedH;
        const float combined=h->score+0.12f*item.samScore+0.06f*item.dinoScore;
        if(!best || combined>bestCombined){
            best=*h;
            bestCombined=combined;
            lastDinoScore_=item.dinoScore;
            lastSamScore_=item.samScore;
        }
    }
    return best;
}

// ============================================================================
// V15.38 IDENTITY AUTHORITY
//
// Motion and identity are deliberately separate:
//   - optical flow may predict motion every frame;
//   - only fresh H evidence may renew an authoritative lock;
//   - a stale identity becomes SUSPECT even if optical flow is excellent.
//
// The two background workers below are single-flight + latest-request-only.
// There is no verification backlog and no old-frame catch-up.
// ============================================================================

void HelipadTracker::invalidateIdentityAuthority(){
    ++targetGeneration_;
    haveIdentityProof_=false;
    haveRingProof_=false;
    haveVerifiedTarget_=false;
    verifiedTarget_={};
    cumulativeFlow_={0,0};
    verifiedFlowAccum_={0,0};
    pendingIdentity_.reset();
    pendingRing_.reset();
    lastIdentityProofTime_={};
    lastRingProofTime_={};
    lastIdentityRequestTime_={};
    lastRingRequestTime_={};
}

float HelipadTracker::identityAgeMs(Clock::time_point now) const {
    if(!haveIdentityProof_) return 1.0e9f;
    return static_cast<float>(
        std::chrono::duration<double,std::milli>(now-lastIdentityProofTime_).count());
}

float HelipadTracker::ringAgeMs(Clock::time_point now) const {
    if(!haveRingProof_) return 1.0e9f;
    return static_cast<float>(
        std::chrono::duration<double,std::milli>(now-lastRingProofTime_).count());
}

bool HelipadTracker::strongIdentityProof(const Candidate& c) const {
    // NON-WEAKENING RULE:
    // Do not invent a stricter second threshold after a detector has already
    // returned a valid H.  Each detector keeps its proven acceptance rule.
    // The false-lock fix comes from the wall-clock identity lease: when no H
    // is returned, excellent optical flow can no longer renew identity.
    switch(c.source){
        case CandidateSource::FarH:
            return c.score>=cfg_.h_far_confirm_score;

        case CandidateSource::ObliqueH:
            return c.score>=cfg_.h_oblique_confirm_score;

        case CandidateSource::GenericH:
        case CandidateSource::AiVerifiedH:
            return c.score>=cfg_.universal_local_h_min;

        case CandidateSource::PlainH:
            // Exact existing HMode::Track identity threshold in vision.cpp.
            return c.score>=0.68f;

        case CandidateSource::RingH:
            // Ring geometry is never accepted by the H identity worker.
            // Ring rescue has its own strict two-key gate below.
            return false;

        case CandidateSource::None:
            return false;
    }
    return false;
}

bool HelipadTracker::strongRingIdentityRescue(const Candidate& c) const {
    // Only a strong DOUBLE-ring + strong H may independently rescue identity.
    // Single-ellipse RingH remains useful as classification/support, but it is
    // deliberately not allowed to revive a stale lock by itself.
    return c.hasRing &&
           c.ringPairScore>=0.30f &&
           c.ringFitScore>=0.40f &&
           c.shapeScore>=0.80f &&
           c.topology>=0.70f &&
           c.railMin>=0.18f &&
           c.crossbar>=0.34f &&
           std::min(c.voidTop,c.voidBottom)>=0.58f;
}


// ============================================================================
// V15.39 STRONG OBLIQUE AUTHORITY
//
// This is an accelerator, never a rejection gate. If a candidate does not meet
// these conservative tests, the original V15.38 confirmation path continues.
// ============================================================================

void HelipadTracker::resetFastObliqueProof(){
    lastStrongObliqueProofTime_={};
    lastObliqueFastVerifyTime_={};
    lastStrongObliqueCandidate_={};
    haveStrongObliqueCandidate_=false;
    fastObliqueHits_=0;
    currentConfirmAuthority_=0.0f;
    currentConfirmContrast_=0.0f;
}

float HelipadTracker::robustCandidateContrast(const cv::Mat& gray,
                                               const Candidate& c) const {
    if(gray.empty()) return 0.0f;
    // Use the transported/predicted box first. rawBox is not motion-transported
    // during confirmation, so preferring it would measure stale pixels.
    cv::Rect2f base=c.box.area()>9.0f ? c.box : c.rawBox;
    if(base.area()<=0.0f) return 0.0f;

    cv::Rect2f roi=expandRect(base,1.30f,gray.cols,gray.rows);
    cv::Rect r(roi);
    r &= cv::Rect(0,0,gray.cols,gray.rows);
    if(r.width<5 || r.height<5) return 0.0f;

    const int step=std::max(1,int(std::sqrt(double(r.area())/4096.0)));
    std::vector<uchar> values;
    values.reserve(size_t((r.width/step+1)*(r.height/step+1)));
    for(int y=r.y;y<r.y+r.height;y+=step){
        const uchar* row=gray.ptr<uchar>(y);
        for(int x=r.x;x<r.x+r.width;x+=step) values.push_back(row[x]);
    }
    if(values.size()<16) return 0.0f;

    const size_t n=values.size();
    const size_t i10=std::min(n-1,size_t(std::floor(0.10*double(n-1))));
    const size_t i90=std::min(n-1,size_t(std::floor(0.90*double(n-1))));
    std::nth_element(values.begin(),values.begin()+i10,values.end());
    const float p10=float(values[i10]);
    std::nth_element(values.begin(),values.begin()+i90,values.end());
    const float p90=float(values[i90]);
    return std::max(0.0f,p90-p10);
}

float HelipadTracker::obliqueConfirmationAuthority(const cv::Mat& gray,
                                                    const Candidate& c,
                                                    float* contrastOut) const {
    const float contrast=robustCandidateContrast(gray,c);
    if(contrastOut) *contrastOut=contrast;

    // Hard prerequisites apply only to the fast route. Failure here means
    // "use V15.38", not "reject H".
    if(c.score<cfg_.oblique_fast_min_score ||
       c.topology<cfg_.oblique_fast_min_topology ||
       c.railAvg<cfg_.oblique_fast_min_rail_avg ||
       c.railMin<cfg_.oblique_fast_min_rail ||
       c.crossbar<cfg_.oblique_fast_min_crossbar ||
       std::min(c.voidTop,c.voidBottom)<cfg_.oblique_fast_min_void ||
       contrast<cfg_.oblique_fast_min_contrast)
        return 0.0f;

    const float scoreN=std::clamp((c.score-0.80f)/0.17f,0.0f,1.0f);
    const float topoN=std::clamp((c.topology-0.86f)/0.14f,0.0f,1.0f);
    const float railN=std::clamp((c.railAvg-0.35f)/0.35f,0.0f,1.0f);
    const float railBalance=c.railAvg>1e-4f
        ? std::clamp(c.railMin/c.railAvg,0.0f,1.0f) : 0.0f;
    const float crossN=std::clamp((c.crossbar-0.30f)/0.45f,0.0f,1.0f);
    const float voidN=std::clamp(
        (std::min(c.voidTop,c.voidBottom)-0.45f)/0.45f,0.0f,1.0f);
    const float contrastN=std::clamp((contrast-60.0f)/100.0f,0.0f,1.0f);

    return std::clamp(
        0.25f*scoreN + 0.20f*topoN + 0.13f*railN +
        0.10f*railBalance + 0.10f*crossN + 0.07f*voidN +
        0.15f*contrastN,
        0.0f,1.0f);
}

float HelipadTracker::competitionAuthority(const cv::Mat& gray,
                                            const Candidate& c) const {
    float contrast=0.0f;
    if(c.source==CandidateSource::ObliqueH){
        const float strong=obliqueConfirmationAuthority(gray,c,&contrast);
        if(strong>0.0f) return strong;
    }else{
        contrast=robustCandidateContrast(gray,c);
    }

    // Ranking only; never a lock condition.
    const float scoreN=std::clamp((c.score-0.68f)/0.28f,0.0f,1.0f);
    const float topoN=std::clamp((c.topology-0.65f)/0.35f,0.0f,1.0f);
    const float railN=std::clamp((c.railAvg-0.25f)/0.45f,0.0f,1.0f);
    const float contrastN=std::clamp((contrast-35.0f)/125.0f,0.0f,1.0f);
    const float strictBonus=c.source==CandidateSource::PlainH ? 0.08f : 0.0f;
    return std::clamp(0.40f*scoreN+0.25f*topoN+0.17f*railN+
                      0.18f*contrastN+strictBonus,0.0f,1.0f);
}

bool HelipadTracker::updateFastObliqueProof(const cv::Mat& gray,
                                             const Candidate& c,
                                             Clock::time_point now){
    float contrast=0.0f;
    const float authority=obliqueConfirmationAuthority(gray,c,&contrast);
    currentConfirmAuthority_=authority;
    currentConfirmContrast_=contrast;

    if(authority<cfg_.oblique_fast_authority_min){
        if(haveStrongObliqueCandidate_){
            const double ageMs=std::chrono::duration<double,std::milli>(
                now-lastStrongObliqueProofTime_).count();
            if(ageMs>cfg_.oblique_fast_max_gap_ms) resetFastObliqueProof();
        }
        return false;
    }

    if(!haveStrongObliqueCandidate_){
        lastStrongObliqueCandidate_=c;
        lastStrongObliqueProofTime_=now;
        haveStrongObliqueCandidate_=true;
        fastObliqueHits_=1;
        return fastObliqueHits_>=cfg_.oblique_fast_required_hits;
    }

    const double gapMs=std::chrono::duration<double,std::milli>(
        now-lastStrongObliqueProofTime_).count();
    const float prevSpan=std::max(6.0f,rectSpan(lastStrongObliqueCandidate_.box));
    const float newSpan=std::max(6.0f,rectSpan(c.box));
    const float ratio=newSpan/prevSpan;
    const float continuityRadius=std::max(
        18.0f,cfg_.oblique_fast_continuity_factor*std::max(prevSpan,newSpan));
    const bool sameTarget=cv::norm(c.center-lastStrongObliqueCandidate_.center)
        <=continuityRadius;
    const bool scaleOK=ratio>=cfg_.oblique_fast_scale_ratio_min &&
                       ratio<=cfg_.oblique_fast_scale_ratio_max;

    if(gapMs>cfg_.oblique_fast_max_gap_ms || !sameTarget || !scaleOK){
        lastStrongObliqueCandidate_=c;
        lastStrongObliqueProofTime_=now;
        fastObliqueHits_=1;
        return false;
    }

    if(gapMs>=cfg_.oblique_fast_min_gap_ms){
        ++fastObliqueHits_;
        lastStrongObliqueCandidate_=c;
        lastStrongObliqueProofTime_=now;
    }
    return fastObliqueHits_>=cfg_.oblique_fast_required_hits;
}

// ============================================================================
// V15.40 RING-H TEMPORAL AUTHORITY
//
// A fast-ring observation already contains:
//   concentric ellipse geometry + an H returned by the existing H detector.
// Two time-separated observations at the same location, together with fresh
// independent H confirmations, are allowed to confirm RingH without forcing
// DEEP_RING to rediscover the exact same ring several times.
//
// This is an accelerator only.  Missing this path never rejects a candidate.
// ============================================================================

void HelipadTracker::resetFastRingProof(){
    lastStrongRingProofTime_={};
    lastRingFastVerifyTime_={};
    lastRingDeepConfirmTime_={};
    lastStrongRingCandidate_={};
    haveStrongRingCandidate_=false;
    fastRingHits_=0;
    ringIndependentHHits_=0;
    ringIndependentSource_=CandidateSource::PlainH;
}

bool HelipadTracker::updateFastRingProof(const Candidate& c,
                                          Clock::time_point now){
    if(!c.hasRing || c.ringPairScore<cfg_.ring_fast_min_pair)
        return false;

    // runFastRingLane() only returns after the unchanged H verifier accepts
    // an H inside the ellipse pair.  Do not add a second H threshold here.
    if(!haveStrongRingCandidate_){
        haveStrongRingCandidate_=true;
        lastStrongRingCandidate_=c;
        lastStrongRingProofTime_=now;
        fastRingHits_=1;
        return false;
    }

    const double gapMs=std::chrono::duration<double,std::milli>(
        now-lastStrongRingProofTime_).count();
    const float oldSpan=std::max(6.0f,rectSpan(lastStrongRingCandidate_.ringBox));
    const float newSpan=std::max(6.0f,rectSpan(c.ringBox));
    const float ratio=newSpan/oldSpan;
    const float radius=std::max(
        18.0f,cfg_.ring_fast_continuity_factor*std::max(oldSpan,newSpan));
    const bool same=cv::norm(c.ringCenter-lastStrongRingCandidate_.ringCenter)<=radius;
    const bool scaleOK=ratio>=cfg_.ring_fast_scale_ratio_min &&
                       ratio<=cfg_.ring_fast_scale_ratio_max;

    if(gapMs>cfg_.ring_fast_max_gap_ms || !same || !scaleOK){
        lastStrongRingCandidate_=c;
        lastStrongRingProofTime_=now;
        fastRingHits_=1;
        return false;
    }

    if(gapMs>=cfg_.ring_fast_min_gap_ms){
        ++fastRingHits_;
        lastStrongRingCandidate_=c;
        lastStrongRingProofTime_=now;
    }

    return fastRingHits_>=cfg_.ring_fast_required_hits;
}

void HelipadTracker::markIdentityProof(const Candidate& proof, Clock::time_point proofTime){
    haveIdentityProof_=true;
    lastIdentityProofTime_=proofTime;
    verifiedTarget_=proof;
    haveVerifiedTarget_=true;
    verifiedFlowAccum_=cumulativeFlow_;
    identityMisses_=0;
}

void HelipadTracker::markRingProof(const Candidate& ring, Clock::time_point proofTime){
    haveRingProof_=true;
    lastRingProofTime_=proofTime;
    rememberRingSupport(ring);
}

void HelipadTracker::launchIdentityRequest(IdentityRequest request){
    identityFuture_=std::async(
        std::launch::async,
        [this, request=std::move(request)]() mutable -> IdentityResult {
            IdentityResult out;
            out.request=request;

            const Candidate& t=request.targetSnapshot;
            std::optional<Candidate> h;

            // V15.42: the background identity worker is source-independent.
            // FAR / Plain / Oblique are alternative ways to prove the SAME H.
            // A detector origin never selects a different ownership state machine.
            if(request.recovery){
                // 1) Current motion prediction: cheapest strict/local proof first.
                h=identityDetector_.findH(
                    request.frame,
                    t.box,
                    t.center,
                    t.box,
                    HMode::Recovery);

                // 2) Last detector-proven H is an additional RECOVERY PRIOR only.
                // queueIdentityVerification() may motion-propagate this prior to
                // the captured frame; that propagation never becomes proof itself.
                if(!h && request.haveVerifiedSnapshot){
                    h=identityDetector_.findH(
                        request.frame,
                        request.verifiedSnapshot.box,
                        std::nullopt,
                        std::nullopt,
                        HMode::Recovery);
                }

                // 3) Perspective rescue, still proving H rather than a new identity.
                if(!h && cfg_.h_oblique_enabled && t.span>=16.0f){
                    h=identityDetector_.findOblique(
                        request.frame,
                        t.center,
                        t.box);
                }

                if(!h && cfg_.h_oblique_enabled &&
                   request.haveVerifiedSnapshot &&
                   request.verifiedSnapshot.span>=16.0f){
                    h=identityDetector_.findOblique(
                        request.frame,
                        request.verifiedSnapshot.center,
                        request.verifiedSnapshot.box);
                }

                // 4) Color-invariant local rescue remains additive.
                if(!h && cfg_.universal_color_enabled){
                    h=identityDetector_.findColorInvariant(
                        request.frame,
                        t.box,
                        t.center,
                        t.box,
                        HMode::Recovery);
                }

                if(!h && cfg_.universal_color_enabled &&
                   request.haveVerifiedSnapshot){
                    h=identityDetector_.findColorInvariant(
                        request.frame,
                        request.verifiedSnapshot.box,
                        std::nullopt,
                        std::nullopt,
                        HMode::Recovery);
                }

                // 5) Tiny-H FAR rescue is last because it is the broadest search.
                if(!h && cfg_.h_far_enabled && t.span<=135.0f &&
                   !cfg_.h_far_scale_sides.empty()){
                    h=identityDetector_.findFar(
                        request.frame,
                        t.center,
                        t.box,
                        cfg_.h_far_scale_sides.front());
                }
            }else{
                // Healthy lock: strict/local H first on every identity job.
                h=identityDetector_.findH(
                    request.frame,
                    t.box,
                    t.center,
                    t.box,
                    HMode::Track);

                if(!h && cfg_.h_oblique_enabled && t.span>=16.0f){
                    h=identityDetector_.findOblique(
                        request.frame,
                        t.center,
                        t.box);
                }

                if(!h && cfg_.universal_color_enabled){
                    h=identityDetector_.findColorInvariant(
                        request.frame,
                        t.box,
                        t.center,
                        t.box,
                        HMode::Track);
                }

                if(!h && cfg_.h_far_enabled && t.span<=135.0f &&
                   !cfg_.h_far_scale_sides.empty()){
                    h=identityDetector_.findFar(
                        request.frame,
                        t.center,
                        t.box,
                        cfg_.h_far_scale_sides.front());
                }
            }

            out.h=std::move(h);
            return out;
        });
}

void HelipadTracker::launchRingRequest(RingRequest request){
    ringFuture_=std::async(
        std::launch::async,
        [this, request=std::move(request)]() mutable -> RingResult {
            RingResult out;
            out.request=request;

            // Ring is supporting evidence, not primary recovery authority.
            // During SUSPECT recovery never union a potentially drifted flow box
            // with the last verified H: that can create a huge, expensive ROI and
            // reintroduce runway false positives.  Prefer known ring support, then
            // the flow-propagated last verified H, otherwise the current target.
            cv::Rect2f anchor=request.targetSnapshot.box;
            if(request.recovery){
                if(request.haveRingSupport)
                    anchor=request.ringSupportBox;
                else if(request.haveVerifiedSnapshot)
                    anchor=request.verifiedSnapshot.box;
            }else if(request.haveRingSupport){
                anchor=unionRect(anchor,request.ringSupportBox,
                                 request.frame.cols,request.frame.rows);
            }

            out.ring=ringHelper_.findRingH(
                request.frame,
                anchor,
                request.recovery
                    ? std::optional<cv::Point2f>{}
                    : std::optional<cv::Point2f>(request.targetSnapshot.center),
                request.recovery
                    ? std::optional<cv::Rect2f>{}
                    : std::optional<cv::Rect2f>(request.targetSnapshot.box),
                request.recovery ? HMode::Recovery : HMode::Track);

            return out;
        });
}

void HelipadTracker::queueIdentityVerification(const cv::Mat& bgr,
                                                Clock::time_point now,
                                                bool recovery){
    if(bgr.empty() || target_.source==CandidateSource::None) return;

    IdentityRequest request;
    request.frame=bgr.clone();
    request.targetSnapshot=target_;
    request.verifiedSnapshot=verifiedTarget_;
    request.haveVerifiedSnapshot=haveVerifiedTarget_;
    if(request.haveVerifiedSnapshot){
        // Propagate the last REAL H proof with optical motion up to this frame.
        // The propagated position is only a recovery prior; it does not renew
        // the identity lease.
        const cv::Point2f vd=cumulativeFlow_-verifiedFlowAccum_;
        request.verifiedSnapshot.center+=vd;
        request.verifiedSnapshot.box=clampRect(
            moveRect(request.verifiedSnapshot.box,vd.x,vd.y),bgr.cols,bgr.rows);
        request.verifiedSnapshot.rawBox=clampRect(
            moveRect(request.verifiedSnapshot.rawBox,vd.x,vd.y),bgr.cols,bgr.rows);
    }
    request.recovery=recovery;
    request.generation=targetGeneration_;
    request.serial=++identitySerial_;
    request.flowAccumAtSubmit=cumulativeFlow_;
    request.capturedAt=now;
    lastIdentityRequestTime_=now;

    if(!identityFuture_.valid()){
        launchIdentityRequest(std::move(request));
        return;
    }

    // Single latest-frame slot.  Replacing this request never queues work.
    pendingIdentity_=std::move(request);
}

void HelipadTracker::queueRingVerification(const cv::Mat& bgr,
                                            Clock::time_point now,
                                            bool recovery){
    if(bgr.empty() || target_.source==CandidateSource::None ||
       !cfg_.universal_ring_enabled) return;

    RingRequest request;
    request.frame=bgr.clone();
    request.targetSnapshot=target_;
    request.verifiedSnapshot=verifiedTarget_;
    request.haveVerifiedSnapshot=haveVerifiedTarget_;
    if(request.haveVerifiedSnapshot){
        // Propagate the last REAL H proof with optical motion up to this frame.
        // The propagated position is only a recovery prior; it does not renew
        // the identity lease.
        const cv::Point2f vd=cumulativeFlow_-verifiedFlowAccum_;
        request.verifiedSnapshot.center+=vd;
        request.verifiedSnapshot.box=clampRect(
            moveRect(request.verifiedSnapshot.box,vd.x,vd.y),bgr.cols,bgr.rows);
        request.verifiedSnapshot.rawBox=clampRect(
            moveRect(request.verifiedSnapshot.rawBox,vd.x,vd.y),bgr.cols,bgr.rows);
    }
    request.recovery=recovery;
    request.haveRingSupport=hasRingSupport_;
    request.ringSupportBox=ringSupportBox_;
    request.generation=targetGeneration_;
    request.serial=++ringSerial_;
    request.flowAccumAtSubmit=cumulativeFlow_;
    request.capturedAt=now;

    if(!ringFuture_.valid()){
        lastRingRequestTime_=now;
        launchRingRequest(std::move(request));
        return;
    }

    // Healthy RingH classification must never become a continuous CPU burner
    // when one ring check takes longer than its nominal period.  Only SUSPECT
    // recovery keeps one latest pending ring request; healthy LOCK simply waits
    // for the current ring job to finish and then observes a fresh cooldown.
    if(recovery){
        pendingRing_=std::move(request);
        lastRingRequestTime_=now;
    }
}

bool HelipadTracker::pollIdentityVerification(const cv::Mat& gray,
                                               Clock::time_point now){
    if(!identityFuture_.valid()) return false;
    if(identityFuture_.wait_for(std::chrono::milliseconds(0))!=
       std::future_status::ready) return false;

    IdentityResult out;
    bool gotResult=true;
    try{
        out=identityFuture_.get();
    }catch(...){
        gotResult=false;
        ++identityMisses_;
    }

    // Single latest-frame semantics: never launch a waiting frame that is
    // already too old to become useful identity evidence.
    if(pendingIdentity_){
        IdentityRequest next=std::move(*pendingIdentity_);
        pendingIdentity_.reset();
        const float pendingAgeMs=static_cast<float>(
            std::chrono::duration<double,std::milli>(now-next.capturedAt).count());
        if(next.generation==targetGeneration_ &&
           pendingAgeMs<=cfg_.identity_result_max_age_ms){
            launchIdentityRequest(std::move(next));
        }else{
            lastIdentityRequestTime_={};
        }
    }

    if(!gotResult) return false;
    if(out.request.generation!=targetGeneration_) return false;

    const float resultAgeMs=static_cast<float>(
        std::chrono::duration<double,std::milli>(now-out.request.capturedAt).count());
    if(resultAgeMs>cfg_.identity_result_max_age_ms) return false;

    if(!out.h || !strongIdentityProof(*out.h)){
        ++identityMisses_;
        currentHScore_*=0.97f;
        return false;
    }

    const Candidate& h=*out.h;
    const float snapSpan=std::max(6.0f,rectSpan(out.request.targetSnapshot.box));
    const float hSpan=std::max(4.0f,rectSpan(h.box));
    const float scaleRatio=hSpan/snapSpan;
    const float dCurrent=cv::norm(h.center-out.request.targetSnapshot.center);

    bool spatialOK=false;
    bool nearCurrent=false;
    bool nearVerified=false;

    if(out.request.recovery){
        nearCurrent=dCurrent<=std::max(90.0f,4.0f*snapSpan);
        if(out.request.haveVerifiedSnapshot){
            const float vs=std::max(
                6.0f,rectSpan(out.request.verifiedSnapshot.box));
            nearVerified=
                cv::norm(h.center-out.request.verifiedSnapshot.center)<=
                std::max(110.0f,4.5f*vs);
        }
        spatialOK=(nearCurrent || nearVerified) &&
                  scaleRatio>=0.40f && scaleRatio<=2.40f;
    }else{
        nearCurrent=dCurrent<=std::max(8.0f,0.72f*snapSpan);
        spatialOK=nearCurrent &&
                  scaleRatio>=0.58f && scaleRatio<=1.65f;
    }

    if(!spatialOK){
        ++identityMisses_;
        currentHScore_*=0.97f;
        return false;
    }

    // ====================================================================
    // V15.42 IMMUTABLE ASYNC IDENTITY
    //
    // `h` is detector truth in out.request.frame.  Its pixel coordinates are
    // never translated by optical flow and never written into the current
    // target position.  Optical flow remains a motion estimate only.
    // ====================================================================
    const bool wasSuspect=(state_==TrackState::Suspect) || out.request.recovery;
    currentHScore_=wasSuspect
        ? h.score
        : 0.72f*currentHScore_+0.28f*h.score;

    const HAcquisitionMethod proofMethod=methodFromSource(h.source);
    if(proofMethod!=HAcquisitionMethod::None &&
       proofMethod!=HAcquisitionMethod::RingRoi){
        acquisitionMethod_=proofMethod;
        lastHSource_=canonicalHSource(proofMethod);
    }

    // This stores the exact detector-backed coordinates and the exact capture
    // timestamp.  The flow accumulator is then set to the accumulator that
    // belonged to that captured frame, so a later recovery request may form a
    // motion PRIOR without rewriting detector truth.
    markIdentityProof(h,out.request.capturedAt);
    verifiedFlowAccum_=out.request.flowAccumAtSubmit;

    const float proofAge=identityAgeMs(now);
    if(proofAge<=cfg_.identity_soft_lease_ms){
        state_=TrackState::Locked;
        if(!flow_.ready())
            flow_.seed(gray,target_.box);
    }

    return true;
}

bool HelipadTracker::pollRingVerification(const cv::Mat& gray,
                                           Clock::time_point now){
    (void)gray;
    if(!ringFuture_.valid()) return false;
    if(ringFuture_.wait_for(std::chrono::milliseconds(0))!=
       std::future_status::ready) return false;

    RingResult out;
    bool gotResult=true;
    try{
        out=ringFuture_.get();
    }catch(...){
        gotResult=false;
        ++ringSupportMisses_;
    }

    // A completed expensive ring check starts a fresh cooldown.  Stale pending
    // recovery work is discarded rather than becoming a backlog.
    lastRingRequestTime_=now;
    if(pendingRing_){
        RingRequest next=std::move(*pendingRing_);
        pendingRing_.reset();
        const float pendingAgeMs=static_cast<float>(
            std::chrono::duration<double,std::milli>(now-next.capturedAt).count());
        if(next.generation==targetGeneration_ &&
           pendingAgeMs<=std::max(cfg_.identity_result_max_age_ms,900)){
            launchRingRequest(std::move(next));
        }
    }

    if(!gotResult) return false;
    if(out.request.generation!=targetGeneration_) return false;

    const float resultAgeMs=static_cast<float>(
        std::chrono::duration<double,std::milli>(now-out.request.capturedAt).count());
    if(resultAgeMs>std::max(cfg_.identity_result_max_age_ms,900)) return false;

    if(!out.ring || !out.ring->hasRing){
        ++ringSupportMisses_;
        return false;
    }

    const Candidate& ring=*out.ring;
    const float snapSpan=std::max(6.0f,rectSpan(out.request.targetSnapshot.box));
    const float ringSpan=std::max(4.0f,rectSpan(ring.box));
    const float scaleRatio=ringSpan/snapSpan;
    const float dist=cv::norm(ring.center-out.request.targetSnapshot.center);

    // The strict RingHelper result is validated in the SAME captured frame as
    // its target snapshot.  No later optical-flow displacement participates.
    const bool associated=
        dist<=std::max(14.0f,0.90f*snapSpan) &&
        scaleRatio>=0.45f && scaleRatio<=2.10f;
    if(!associated){
        ++ringSupportMisses_;
        return false;
    }

    // ====================================================================
    // V15.42 RING SIDECAR
    //
    // Ring work may classify the already-owned H, but it cannot move H and it
    // cannot renew H identity.  Ring loss later expires only TargetClass.
    // ====================================================================
    targetClass_=TargetClass::RingH;
    ringClassHits_=std::max(ringClassHits_,2);
    lastRingClassCandidate_=ring;
    lastRingClassObservationTime_=out.request.capturedAt;
    haveRingClassCandidate_=true;

    haveRingProof_=true;
    lastRingProofTime_=out.request.capturedAt;
    ringSupportMisses_=0;

    return true;
}

// ============================================================================
// V15.56 PRIMARY PUREH_CORELOCK_05 AUTHORITY
//
// PureH_CoreLock_05 owns H acquisition, confirmation, sticky identity,
// direct re-verification, bounded prediction/recovery and challenger switching.
// Aryan's TensorRT AI worker is intentionally unchanged: it may nominate a
// current-frame ROI, but CoreLock05 must independently verify that ROI and still
// pass its normal multi-observation confirmation before identity can exist.
// Ring remains a classification sidecar and never renews H identity.
// ============================================================================
TrackResult HelipadTracker::processCoreLock05(
    const cv::Mat& bgr,
    Clock::time_point now)
{
    TrackResult result;
    if(bgr.empty() || !coreLock05_){
        result.status=bgr.empty()?"EMPTY FRAME":"CORE5 UNAVAILABLE";
        return result;
    }

    cv::Mat gray;
    cv::cvtColor(bgr,gray,cv::COLOR_BGR2GRAY);

    // V15.57 HYBRID PROPOSAL BUS + PUREH AUTHORITY
    //
    // Keep Aryan's broad proposal coverage (PLAIN/FAR/OBLIQUE/COLOR) but do
    // NOT let those lanes own identity. Each proposal is independently
    // re-verified by PureH_CoreLock_05 and then must pass CoreLock's normal
    // multi-observation confirmation. PureH also keeps its own internal search.
    // This restores the proposal coverage accidentally bypassed by V15.56.
    //
    // AI is intentionally untouched: pollAiEvidence() is the exact existing
    // V15.55/V15.56 function and remains ROI nomination only.
    std::optional<Candidate> aiCurrentProof=pollAiEvidence(bgr,now);

    bool geometryHintAccepted=false;

    auto hintLaneFromSource=[](CandidateSource source){
        switch(source){
            case CandidateSource::FarH: return CoreLock05HintLane::Far;
            case CandidateSource::ObliqueH: return CoreLock05HintLane::Oblique;
            case CandidateSource::GenericH: return CoreLock05HintLane::Generic;
            case CandidateSource::PlainH: return CoreLock05HintLane::Plain;
            default: return CoreLock05HintLane::None;
        }
    };

    auto offerGeometryCandidate=[&](const Candidate& c){
        const CoreLock05HintLane lane=hintLaneFromSource(c.source);
        if(lane==CoreLock05HintLane::None) return false;
        const cv::Rect2f support=c.rawBox.area()>0.0f?c.rawBox:c.box;
        return coreLock05_->offerGeometryHint(
            bgr,c.center,support,lane);
    };

    if(coreLock05_->wantsExternalProposal()){
        if(searchStartFrame_<=0) searchStartFrame_=frame_;
        if(acquisitionStartFrame_<=0) acquisitionStartFrame_=frame_;

        const int acqAge=frame_-acquisitionStartFrame_;
        const bool farDue=
            cfg_.h_far_enabled &&
            !cfg_.h_far_scale_sides.empty() &&
            frame_-lastFarHFrame_>=cfg_.h_far_search_interval;
        const bool directDue=
            frame_-lastGlobalHFrame_>=cfg_.h_global_search_interval ||
            (cfg_.h_oblique_enabled &&
             frame_-lastObliqueHFrame_>=cfg_.h_oblique_search_interval);

        auto chooseDirect=[&](const std::optional<Candidate>& normal,
                              const std::optional<Candidate>& oblique)
            -> std::optional<Candidate> {
            if(!normal) return oblique;
            if(!oblique) return normal;
            const float d=cv::norm(normal->center-oblique->center);
            const float sameRadius=std::max(
                45.0f,0.55f*std::max(normal->span,oblique->span));
            if(d<=sameRadius){
                Candidate q=*normal;
                if(oblique->span>normal->span){
                    q.box=oblique->box;
                    q.rawBox=oblique->rawBox;
                    q.span=oblique->span;
                }
                q.center=0.6f*normal->center+0.4f*oblique->center;
                q.score=std::max(normal->score,oblique->score);
                q.source=normal->source;
                return q;
            }
            if(oblique->span>=1.25f*normal->span &&
               oblique->score>=cfg_.h_oblique_acquire_score)
                return oblique;
            return normal;
        };

        auto runDirectProposal=[&]() -> std::optional<Candidate> {
            std::optional<Candidate> normal;
            std::optional<Candidate> oblique;
            const bool doNormal=
                frame_-lastGlobalHFrame_>=cfg_.h_global_search_interval;
            const bool doOblique=cfg_.h_oblique_enabled &&
                frame_-lastObliqueHFrame_>=cfg_.h_oblique_search_interval;

            if(doNormal && doOblique){
                auto fNormal=std::async(std::launch::async,[&](){
                    return hDetector_.findH(
                        bgr,std::nullopt,std::nullopt,std::nullopt,HMode::Acquire);
                });
                oblique=hDetector_.findOblique(bgr);
                normal=fNormal.get();
                lastGlobalHFrame_=frame_;
                lastObliqueHFrame_=frame_;
            }else{
                if(doNormal){
                    normal=hDetector_.findH(
                        bgr,std::nullopt,std::nullopt,std::nullopt,HMode::Acquire);
                    lastGlobalHFrame_=frame_;
                }
                if(doOblique){
                    oblique=hDetector_.findOblique(bgr);
                    lastObliqueHFrame_=frame_;
                }
            }

            // Keep the V15.43 top-sky false-oblique guard. It affects only the
            // permissive OBLIQUE proposal; Plain/FAR are untouched.
            if(oblique){
                const float topGuardY=std::max(64.0f,0.18f*float(bgr.rows));
                const float obliqueSpan=
                    std::max(oblique->span,rectSpan(oblique->box));
                if(obliqueSpan>30.0f && oblique->center.y<topGuardY)
                    oblique.reset();
            }

            std::optional<Candidate> direct=chooseDirect(normal,oblique);

            // COLOR_H is a geometry fallback, never AI. Run it only when the
            // regular direct family misses, so it increases recall without
            // adding cost to successful Plain/Oblique frames.
            if(!direct && cfg_.universal_color_enabled){
                direct=hDetector_.findColorInvariant(
                    bgr,std::nullopt,std::nullopt,std::nullopt,HMode::Acquire);
            }
            return direct;
        };

        std::optional<Candidate> proposal;
        bool hLaneUsed=false;

        auto tryFar=[&](){
            if(!farDue || hLaneUsed) return;
            hLaneUsed=true;
            const int side=cfg_.h_far_scale_sides[
                farScaleCursor_++%cfg_.h_far_scale_sides.size()];
            proposal=hDetector_.findFar(
                bgr,std::nullopt,std::nullopt,side);
            lastFarHFrame_=frame_;
        };

        auto tryDirect=[&](){
            if(!directDue || hLaneUsed) return;
            hLaneUsed=true;
            proposal=runDirectProposal();
        };

        // Preserve the proven V15.42 acquisition budget: one heavy Aryan H
        // proposal lane per frame. FAR gets 2/3 of initial slots, then FAR and
        // direct alternate. CoreLock performs the independent verification.
        const bool preferFar=cfg_.h_far_enabled &&
            (acqAge<cfg_.h_far_grace_frames
                ? (frame_%3)!=2
                : (frame_%2)==0);

        if(preferFar){
            tryFar();
            if(!hLaneUsed) tryDirect();
        }else{
            tryDirect();
            if(!hLaneUsed) tryFar();
        }

        if(proposal)
            geometryHintAccepted=offerGeometryCandidate(*proposal);
    }

    // AI remains a parallel assistant. It is only offered when a classical
    // geometry proposal did not already enter CoreLock confirmation this frame.
    if(!geometryHintAccepted && aiCurrentProof && coreLock05_->wantsExternalProposal())
        (void)coreLock05_->offerAiHint(bgr,aiCurrentProof->box);

    const CoreLock05PrimaryResult core=coreLock05_->process(bgr);
    const bool hadOwned=
        target_.source!=CandidateSource::None &&
        state_!=TrackState::Searching;

    auto resetRingClassification=[&](){
        targetClass_=TargetClass::PlainH;
        ringClassHits_=0;
        lastRingClassObservationTime_={};
        lastRingClassCandidate_={};
        haveRingClassCandidate_=false;
        hasRingSupport_=false;
        ringSupportMisses_=0;
        haveRingProof_=false;
        lastRingProofTime_={};
    };

    if(core.owned){
        Candidate next;
        next.box=clampRect(core.box,bgr.cols,bgr.rows);
        next.rawBox=next.box;
        next.center=core.center;
        next.score=std::clamp(core.confidence,0.0f,1.0f);
        next.shapeScore=std::clamp(core.shapeScore,0.0f,1.0f);
        next.topology=std::clamp(core.topologyScore,0.0f,1.0f);
        next.span=std::max(core.span,rectSpan(next.box));
        next.area=next.box.area();
        switch(core.sourceLane){
            case CoreLock05HintLane::Far:
                next.source=CandidateSource::FarH; break;
            case CoreLock05HintLane::Oblique:
                next.source=CandidateSource::ObliqueH; break;
            case CoreLock05HintLane::Generic:
                next.source=CandidateSource::GenericH; break;
            case CoreLock05HintLane::Ai:
                next.source=CandidateSource::AiVerifiedH; break;
            case CoreLock05HintLane::Plain:
                next.source=CandidateSource::PlainH; break;
            case CoreLock05HintLane::PureH:
            case CoreLock05HintLane::None:
            default:
                next.source=core.farMode
                    ?CandidateSource::FarH
                    :CandidateSource::PlainH;
                break;
        }

        bool newIdentity=!hadOwned;
        if(hadOwned && core.observedThisFrame){
            const float oldSpan=std::max(6.0f,rectSpan(target_.box));
            const float newSpan=std::max(6.0f,rectSpan(next.box));
            const float ratio=newSpan/oldSpan;
            const float d=cv::norm(next.center-target_.center);
            const float jumpLimit=std::max(120.0f,3.5f*std::max(oldSpan,newSpan));
            // CoreLock already applies challenger hysteresis. This extra test is
            // only for invalidating stale AI/ring generations after a genuine
            // CoreLock identity relocation; it does not reject the CoreLock H.
            if(d>jumpLimit || ratio<0.28f || ratio>3.6f)
                newIdentity=true;
        }

        if(newIdentity){
            invalidateIdentityAuthority();
            resetRingClassification();
            confirmationFailures_=0;
        }

        target_=next;
        acquisitionMethod_=methodFromSource(next.source);
        if(acquisitionMethod_==HAcquisitionMethod::None)
            acquisitionMethod_=core.farMode
                ?HAcquisitionMethod::Far
                :HAcquisitionMethod::Plain;
        lastHSource_=next.source;
        searchStartFrame_=0;
        acquisitionStartFrame_=0;
        currentHScore_=next.score;
        confirmHits_=core.confirm;
        identityMisses_=core.identityMiss;
        lostFrames_=core.lost;
        farConfirmHits_=(next.source==CandidateSource::FarH || core.farMode)
            ?core.confirm:0;
        obliqueConfirmHits_=next.source==CandidateSource::ObliqueH
            ?core.confirm:0;
        globalConfirmHits_=(next.source==CandidateSource::PlainH ||
                            next.source==CandidateSource::GenericH)
            ?core.confirm:0;

        // Only a fresh CoreLock OBSERVED frame renews detector-backed identity.
        // PREDICTED/StrictTracker/frame-motion continuation never calls this.
        if(core.observedThisFrame)
            markIdentityProof(target_,now);

        if(core.state==CoreLock05PrimaryResult::State::Recover &&
           !core.guidanceValid)
            state_=TrackState::Suspect;
        else
            state_=TrackState::Locked;
    }else{
        if(hadOwned || state_!=TrackState::Searching){
            invalidateIdentityAuthority();
            resetRingClassification();
        }
        target_={};
        state_=TrackState::Searching;
        acquisitionMethod_=HAcquisitionMethod::None;
        lastHSource_=CandidateSource::PlainH;
        currentHScore_=0.0f;
        confirmHits_=core.confirm;
        identityMisses_=core.identityMiss;
        lostFrames_=core.lost;
        farConfirmHits_=0;
        obliqueConfirmHits_=0;
        if(hadOwned){
            searchStartFrame_=frame_;
            acquisitionStartFrame_=frame_;
        }
    }

    // Drain/apply the existing asynchronous ring sidecar. Generation checks in
    // pollRingVerification() prevent an old ring result from attaching to a new H.
    const bool ringUpdated=pollRingVerification(gray,now);
    if(ringUpdated && haveRingClassCandidate_)
        rememberRingSupport(lastRingClassCandidate_);

    if(core.owned && cfg_.universal_ring_enabled){
        const bool recovery=state_==TrackState::Suspect;
        const int periodMs=recovery
            ? std::max(350,cfg_.locked_ring_period_ms/2)
            : cfg_.locked_ring_period_ms;
        const double elapsed=lastRingRequestTime_.time_since_epoch().count()==0
            ? 1.0e9
            : std::chrono::duration<double,std::milli>(
                  now-lastRingRequestTime_).count();
        if(elapsed>=periodMs)
            queueRingVerification(bgr,now,recovery);
    }

    // Preserve current ring classification only while its original lease is
    // fresh. Ring never moves the CoreLock target and never renews H identity.
    const bool ringClassFresh=
        core.owned &&
        targetClass_==TargetClass::RingH &&
        haveRingProof_ &&
        ringAgeMs(now)<=cfg_.ring_label_lease_ms;
    if(ringClassFresh && haveRingClassCandidate_){
        target_.hasRing=true;
        target_.ring=lastRingClassCandidate_.ring;
        target_.ringBox=lastRingClassCandidate_.ringBox;
        target_.ringCenter=lastRingClassCandidate_.ringCenter;
        target_.ringPairScore=lastRingClassCandidate_.ringPairScore;
        target_.ringFitScore=lastRingClassCandidate_.ringFitScore;
    }else if(core.owned){
        target_.hasRing=false;
        if(targetClass_==TargetClass::RingH &&
           (!haveRingProof_ || ringAgeMs(now)>cfg_.ring_label_lease_ms)){
            targetClass_=TargetClass::PlainH;
            haveRingClassCandidate_=false;
            hasRingSupport_=false;
            haveRingProof_=false;
        }
    }

    // IMPORTANT: unchanged TensorRT scheduling. Search uses the full frame;
    // CoreLock-owned targets use the same bounded target ROI mechanism as before.
    submitAiRequest(bgr,now);

    Candidate outputTarget=target_;
    if(core.owned){
        outputTarget.source=ringClassFresh
            ? CandidateSource::RingH
            : lastHSource_;
        if(!ringClassFresh) outputTarget.hasRing=false;
    }

    result.state=state_;
    result.identityAgeMs=haveIdentityProof_?identityAgeMs(now):-1.0f;
    result.ringAgeMs=haveRingProof_?ringAgeMs(now):-1.0f;

    // CoreLock05 is now the authority for whether predicted coordinates are
    // still safe for guidance. Fresh OBSERVED frames remain distinguishable
    // through identityAgeMs and the CORE5 status string.
    result.identityValid=core.owned && core.guidanceValid;
    result.guidanceValid=result.identityValid;
    result.hasTarget=core.owned && core.guidanceValid;
    result.hasPrediction=core.owned && !core.guidanceValid;

    if(result.hasTarget || result.hasPrediction)
        result.target=outputTarget;
    if(result.guidanceValid)
        result.guidance=calculateGuidance(core.center,bgr.cols,bgr.rows,cfg_);

    result.flowQuality=0.0f; // CoreLock owns motion continuity internally.
    result.identityMisses=core.identityMiss;
    result.flowLostFrames=core.lost;
    result.confirmHits=core.confirm;
    result.globalConfirmHits=0;
    result.farConfirmHits=core.farMode?core.confirm:0;
    result.obliqueConfirmHits=0;
    result.confirmationFailures=confirmationFailures_;
    result.confirmAuthority=core.confidence;
    result.confirmContrast=0.0f;
    result.fastObliqueHits=0;
    result.fastRingHits=ringClassHits_;
    result.ringIndependentHHits=0;
    result.detectorRan=true;

    // Exact V15.55 AI telemetry: inference state and scores are reported but
    // cannot create identity without CoreLock verification.
    result.gdinoOn=aiEnabled_.load(std::memory_order_acquire) && aiDinoReady_;
    result.sam2On=aiEnabled_.load(std::memory_order_acquire) && aiSamReady_;
    result.aiFps=aiFps_.load(std::memory_order_relaxed);
    result.aiResultAgeMs=lastAiEvidenceCaptureTime_.time_since_epoch().count()==0
        ? -1.0f
        : static_cast<float>(std::chrono::duration<double,std::milli>(
              now-lastAiEvidenceCaptureTime_).count());
    result.gdinoScore=lastDinoScore_;
    result.samScore=lastSamScore_;
    result.aiQueueReplaced=static_cast<int>(std::min<std::size_t>(
        aiRequests_.replacedCount(),static_cast<std::size_t>(2147483647)));
    result.hProofScore=core.confidence;
    if(!core.owned)
        result.hProofMode="NONE";
    else if(core.provisionalFar)
        result.hProofMode="CORE5_FAR_PROVISIONAL";
    else if(core.farMode)
        result.hProofMode="CORE5_FAR";
    else
        result.hProofMode="CORE5_STRICT";
    result.acquisitionMethod=acquisitionMethodName(acquisitionMethod_);
    result.targetClass=targetClassName(targetClass_);

    std::ostringstream ss;
    if(!core.owned){
        ss<<"SEARCHING CORE5/"<<core.searchMode;
    }else if(core.observedThisFrame){
        ss<<"LOCKED CORE5/OBSERVED";
    }else if(core.guidanceValid){
        ss<<"LOCKED CORE5/PREDICT";
    }else{
        ss<<"SUSPECT CORE5/RECOVER";
    }
    if(core.owned)
        ss<<" "<<sourceName(outputTarget.source)
          <<" score="<<core.confidence;
    if(!core.searchMode.empty()) ss<<" mode="<<core.searchMode;
    result.status=ss.str();

    return result;
}

TrackResult HelipadTracker::process(const cv::Mat& bgr){
    ++frame_;
    TrackResult result;
    bool detectorRan=false;
    const auto now=Clock::now();
    float effectiveIdentitySoftLeaseMs=
        static_cast<float>(cfg_.identity_soft_lease_ms);

    if(bgr.empty()){result.status="EMPTY FRAME";return result;}

    // V15.56: CoreLock05 is the primary classical H/identity/lock backend.
    // The legacy Aryan state machine remains below as a rollback/reference path
    // and is not deleted, but normal runtime does not mix the two authorities.
    if(coreLock05_)
        return processCoreLock05(bgr,now);

    cv::Mat gray;
    cv::cvtColor(bgr,gray,cv::COLOR_BGR2GRAY);

    // Poll is O(1) unless a completed AI batch exists. Only then do we spend
    // current-frame CPU on the bounded H-PROOF ROI nominated by AI.
    std::optional<Candidate> aiCurrentProof=pollAiEvidence(bgr,now);

    // V15.42 DRAIN OLD GENERATION WORKERS
    // clearTarget()/seedCandidate() invalidate authority immediately, but
    // std::async work already running cannot be cancelled. Once such a job is
    // ready, consume and discard it outside LOCKED/SUSPECT so it cannot remain
    // a live future that conflicts with later recovery/escalation work.
    if(state_==TrackState::Searching || state_==TrackState::Confirming){
        if(identityFuture_.valid() &&
           identityFuture_.wait_for(std::chrono::milliseconds(0))==
               std::future_status::ready){
            try{ (void)identityFuture_.get(); }catch(...){}
            pendingIdentity_.reset();
        }
        if(ringFuture_.valid() &&
           ringFuture_.wait_for(std::chrono::milliseconds(0))==
               std::future_status::ready){
            try{ (void)ringFuture_.get(); }catch(...){}
            pendingRing_.reset();
        }
    }

    auto chooseDirect = [&](const std::optional<Candidate>& normal,
                            const std::optional<Candidate>& oblique)
        -> std::optional<Candidate> {
        if(!normal) return oblique;
        if(!oblique) return normal;
        const float d=cv::norm(normal->center-oblique->center);
        const float sameRadius=std::max(45.0f,0.55f*std::max(normal->span,oblique->span));
        if(d<=sameRadius){
            // Both detectors describe the same H. Prefer strict V10 identity,
            // but keep the larger box so the complete oblique marker is owned.
            Candidate q=*normal;
            if(oblique->span>normal->span){q.box=oblique->box;q.rawBox=oblique->rawBox;q.span=oblique->span;}
            q.center=0.6f*normal->center+0.4f*oblique->center;
            q.score=std::max(normal->score,oblique->score);
            return q;
        }
        // V14: a much larger, cross-scale-consistent OBLIQUE_H may beat a
        // tiny strict component. This is the exact failure seen in FVC_4_v13.
        if(oblique->span>=1.25f*normal->span &&
           oblique->score>=cfg_.h_oblique_acquire_score)
            return oblique;
        return normal;
    };

    auto runDirectGlobal = [&]() -> std::optional<Candidate> {
        std::optional<Candidate> normal;
        std::optional<Candidate> oblique;
        const bool doNormal=frame_-lastGlobalHFrame_>=cfg_.h_global_search_interval;
        const bool doOblique=cfg_.h_oblique_enabled &&
            frame_-lastObliqueHFrame_>=cfg_.h_oblique_search_interval;

        if(doNormal || doOblique)
            detectorRan=true;
        if(doNormal && doOblique){
            // Independent detectors can run concurrently.  This is only the
            // periodic acquisition/watchdog path; the 60-Hz optical-flow path
            // remains single-threaded and deterministic.
            auto fNormal=std::async(std::launch::async,[&](){
                return hDetector_.findH(
                    bgr,std::nullopt,std::nullopt,std::nullopt,HMode::Acquire);
            });
            oblique=hDetector_.findOblique(bgr);
            normal=fNormal.get();
            lastGlobalHFrame_=frame_;
            lastObliqueHFrame_=frame_;
        }else{
            if(doNormal){
                normal=hDetector_.findH(
                    bgr,std::nullopt,std::nullopt,std::nullopt,HMode::Acquire);
                lastGlobalHFrame_=frame_;
            }
            if(doOblique){
                oblique=hDetector_.findOblique(bgr);
                lastObliqueHFrame_=frame_;
            }
        }
        // V15.43 TOP-EDGE PERMISSIVE GUARD: GLOBAL OBLIQUE
        //
        // Acquisition-only rejection of large sky/cloud Oblique
        // candidates. PlainH and FarH are NOT modified.
        if((state_==TrackState::Searching ||
            state_==TrackState::Confirming) &&
           oblique)
        {
            const float topGuardY =
                std::max(
                    64.0f,
                    0.18f * float(bgr.rows));

            const float obliqueSpan =
                std::max(
                    oblique->span,
                    rectSpan(oblique->box));

            if(obliqueSpan > 30.0f &&
               oblique->center.y < topGuardY)
            {
                oblique.reset();
            }
        }

        return chooseDirect(normal,oblique);
    };


    // ========================================================
    // V15.23 INDEPENDENT DETECTOR LANES
    //
    // FAST RING PROPOSAL LANE
    //
    // This is intentionally independent of:
    //   lastFarHFrame_
    //   lastGlobalHFrame_
    //   lastRingHFrame_
    //
    // Therefore improving FAR_H cannot suppress RingH.
    //
    // The proposal stage does NOT accept targets.
    // V15.40 confirms it only after time-separated same-target ring geometry
    // plus independent H evidence; DEEP RingHelper remains the fallback.
    // ========================================================
    auto runFastRingLane =
        [&]() -> std::optional<Candidate>
    {
        if(!cfg_.universal_ring_enabled ||
           bgr.empty())
            return std::nullopt;

        constexpr int MAX_SIDE = 960;

        const float scale =
            std::min(
                1.0f,
                float(MAX_SIDE) /
                float(
                    std::max(
                        bgr.cols,
                        bgr.rows)));

        cv::Mat work;

        if(scale < 0.999f)
        {
            cv::resize(
                bgr,
                work,
                cv::Size(),
                scale,
                scale,
                cv::INTER_AREA);
        }
        else
        {
            work = bgr;
        }

        cv::Mat grayRing;

        cv::cvtColor(
            work,
            grayRing,
            cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(
            grayRing,
            grayRing,
            cv::Size(5,5),
            0);

        cv::Scalar mean;
        cv::Scalar sigma;

        cv::meanStdDev(
            grayRing,
            mean,
            sigma);

        const int low =
            std::clamp(
                int(mean[0] -
                    0.85*sigma[0]),
                15,
                85);

        const int high =
            std::clamp(
                int(mean[0] +
                    1.35*sigma[0]),
                low+20,
                210);

        cv::Mat edges;

        cv::Canny(
            grayRing,
            edges,
            low,
            high,
            3,
            true);

        std::vector<
            std::vector<cv::Point>
        > contours;

        cv::findContours(
            edges,
            contours,
            cv::RETR_LIST,
            cv::CHAIN_APPROX_NONE);

        struct RingEllipse
        {
            cv::RotatedRect e;
            float area = 0;
            float residual = 1.0f;
        };

        std::vector<RingEllipse> ellipses;

        ellipses.reserve(
            contours.size());

        for(const auto& c : contours)
        {
            if(c.size() < 20)
                continue;

            if(cv::arcLength(
                   c,
                   false) < 24.0)
                continue;

            cv::RotatedRect e;

            try
            {
                e = cv::fitEllipse(c);
            }
            catch(...)
            {
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

            if(mn < 6.0f)
                continue;

            if(mx >
               0.55f *
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
               cfg_.universal_ring_max_aspect +
               0.65f)
                continue;

            // V15.38 fast geometric residual.  This is intentionally bounded
            // (<= ~80 samples/contour), so obviously non-elliptic runway paint
            // is rejected before any H verifier is called.
            const float th=e.angle*float(CV_PI/180.0);
            const float ca=std::cos(th);
            const float sa=std::sin(th);
            const int sampleStep=std::max(1,int(c.size()/80));
            double residualSum=0.0;
            int residualCount=0;
            for(size_t qi=0; qi<c.size(); qi+=size_t(sampleStep)){
                const float x=float(c[qi].x)-e.center.x;
                const float y=float(c[qi].y)-e.center.y;
                const float xr=x*ca+y*sa;
                const float yr=-x*sa+y*ca;
                const float rr=std::sqrt(
                    std::pow(xr/std::max(1.0f,0.5f*e.size.width),2.0f)+
                    std::pow(yr/std::max(1.0f,0.5f*e.size.height),2.0f));
                residualSum+=std::fabs(rr-1.0f);
                ++residualCount;
            }
            const float residual=residualCount>0
                ? float(residualSum/double(residualCount))
                : 1.0f;

            // Keep this guard deliberately loose: fit quality is primarily a
            // ranking feature.  Only gross non-ellipses are removed here so the
            // new cheap lane cannot lose difficult perspective cases that the
            // previous RingHelper path could still confirm.
            if(residual>std::max(
                   0.55f,
                   2.75f*cfg_.universal_ring_fit_residual))
                continue;

            ellipses.push_back({
                e,
                e.size.width * e.size.height,
                residual
            });
        }

        if(ellipses.empty())
            return std::nullopt;


        struct RingProposal
        {
            cv::RotatedRect e;
            float score = 0;
            float pairScore = 0;
            float fitScore = 0;
            float innerRatio = 0;
        };

        std::vector<RingProposal> proposals;


        auto angleDelta =
            [](float a,float b)
        {
            float d =
                std::fabs(a-b);

            while(d >= 180.0f)
                d -= 180.0f;

            return
                std::min(
                    d,
                    180.0f-d);
        };


        // ----------------------------------------
        // Concentric-ellipse proof.
        //
        // Canny normally creates inner + outer
        // edges even for one painted circular ring.
        // ----------------------------------------
        // ========================================================
        // V15.35B SPATIAL RING PAIRING
        //
        // SAME GEOMETRY / SAME THRESHOLDS / SAME RESULT RULES.
        //
        // Previous implementation compared every ellipse to
        // every other ellipse: O(N^2).
        //
        // Existing acceptance rules already prove that a valid
        // concentric pair cannot be farther apart than:
        //
        //   max(6 px, 0.20 * outer_min_axis)
        //
        // and accepted ellipses already satisfy:
        //
        //   outer_min_axis <= outer_max_axis
        //                  <= 0.55 * max(work width,height)
        //
        // Therefore every possible valid pair is guaranteed to
        // lie within:
        //
        //   max(6, 0.11 * max(work width,height))
        //
        // We bin ellipse centres spatially and skip only pairs
        // that are mathematically unable to pass the ORIGINAL
        // centre-distance test.
        //
        // Pair scoring below is otherwise unchanged.
        // ========================================================

        constexpr int RING_CELL = 32;

        const int ringGridW =
            std::max(
                1,
                (work.cols + RING_CELL - 1) /
                RING_CELL);

        const int ringGridH =
            std::max(
                1,
                (work.rows + RING_CELL - 1) /
                RING_CELL);

        std::vector<std::vector<int>> ringBins(
            static_cast<size_t>(
                ringGridW * ringGridH));

        auto ringCellX =
            [&](float x)
        {
            const int c =
                int(std::floor(
                    x / float(RING_CELL)));

            return std::clamp(
                c,
                0,
                ringGridW-1);
        };

        auto ringCellY =
            [&](float y)
        {
            const int c =
                int(std::floor(
                    y / float(RING_CELL)));

            return std::clamp(
                c,
                0,
                ringGridH-1);
        };

        for(size_t k=0;
            k<ellipses.size();
            ++k)
        {
            const int cx =
                ringCellX(
                    ellipses[k].e.center.x);

            const int cy =
                ringCellY(
                    ellipses[k].e.center.y);

            ringBins[
                static_cast<size_t>(
                    cy*ringGridW + cx)]
                .push_back(
                    static_cast<int>(k));
        }

        const float maxPairRadius =
            std::max(
                6.0f,
                0.11f *
                float(
                    std::max(
                        work.cols,
                        work.rows)));

        std::vector<int> nearbyEllipseIndices;

        for(size_t i=0;
            i<ellipses.size();
            ++i)
        {
            const cv::Point2f ci =
                ellipses[i].e.center;

            const int minCellX =
                ringCellX(
                    ci.x-maxPairRadius);

            const int maxCellX =
                ringCellX(
                    ci.x+maxPairRadius);

            const int minCellY =
                ringCellY(
                    ci.y-maxPairRadius);

            const int maxCellY =
                ringCellY(
                    ci.y+maxPairRadius);

            nearbyEllipseIndices.clear();

            for(int cy=minCellY;
                cy<=maxCellY;
                ++cy)
            {
                for(int cx=minCellX;
                    cx<=maxCellX;
                    ++cx)
                {
                    const auto& bin =
                        ringBins[
                            static_cast<size_t>(
                                cy*ringGridW+cx)];

                    for(const int j : bin)
                    {
                        if(j >
                           static_cast<int>(i))
                        {
                            nearbyEllipseIndices
                                .push_back(j);
                        }
                    }
                }
            }

            // Restore original j ordering. This means even
            // stable_sort tie behaviour for proposals remains
            // consistent with the old nested i/j traversal.
            std::sort(
                nearbyEllipseIndices.begin(),
                nearbyEllipseIndices.end());

            for(const int jj :
                nearbyEllipseIndices)
            {
                const size_t j =
                    static_cast<size_t>(jj);

                const RingEllipse* outer =
                    &ellipses[i];

                const RingEllipse* inner =
                    &ellipses[j];

                if(inner->area >
                   outer->area)
                {
                    std::swap(
                        outer,
                        inner);
                }

                const float omin =
                    std::min(
                        outer->e.size.width,
                        outer->e.size.height);

                const float centreDistance =
                    cv::norm(
                        outer->e.center -
                        inner->e.center);

                if(centreDistance >
                   std::max(
                       6.0f,
                       0.20f*omin))
                {
                    continue;
                }

                const float rw =
                    inner->e.size.width /
                    std::max(
                        1.0f,
                        outer->e.size.width);

                const float rh =
                    inner->e.size.height /
                    std::max(
                        1.0f,
                        outer->e.size.height);

                if(rw < 0.42f ||
                   rw > 0.98f ||
                   rh < 0.42f ||
                   rh > 0.98f)
                {
                    continue;
                }

                const float oa =
                    std::max(
                        outer->e.size.width,
                        outer->e.size.height) /
                    std::max(
                        1.0f,
                        std::min(
                            outer->e.size.width,
                            outer->e.size.height));

                const float ia =
                    std::max(
                        inner->e.size.width,
                        inner->e.size.height) /
                    std::max(
                        1.0f,
                        std::min(
                            inner->e.size.width,
                            inner->e.size.height));

                if(std::fabs(oa-ia) >
                   0.75f)
                {
                    continue;
                }

                if(angleDelta(
                       outer->e.angle,
                       inner->e.angle) >
                   30.0f)
                {
                    continue;
                }

                const float align =
                    std::exp(
                        -std::pow(
                            centreDistance /
                            std::max(
                                3.0f,
                                0.15f*omin),
                            2.0f));

                const float symmetry =
                    std::exp(
                        -3.0f *
                        std::fabs(
                            rw-rh));

                const float pairScore =
                    align*symmetry;

                if(pairScore < 0.28f)
                {
                    continue;
                }

                const float fitScore=
                    std::clamp(
                        1.0f-
                        std::max(outer->residual,inner->residual)/
                        std::max(0.01f,2.75f*cfg_.universal_ring_fit_residual),
                        0.0f,
                        1.0f);

                const float innerRatio=0.5f*(rw+rh);
                const float geometryScore=
                    0.72f*pairScore +
                    0.28f*fitScore;

                proposals.push_back({
                    outer->e,
                    geometryScore,
                    pairScore,
                    fitScore,
                    innerRatio
                });
            }
        }

        if(proposals.empty())
            return std::nullopt;


        std::stable_sort(
            proposals.begin(),
            proposals.end(),
            [](const RingProposal& a,
               const RingProposal& b)
            {
                return
                    a.score >
                    b.score;
            });


        // ----------------------------------------
        // Verify only the best few ring proposals.  Unlike V15.35, the
        // verification below is a SMALL-ROI H check, not a nested RingHelper
        // scan, so a third proposal improves recall without recreating the old
        // full-frame cost.
        //
        // This is the critical difference from the
        // old full-frame RingHelper:
        //
        // hundreds of random ellipses do NOT receive
        // hundreds of H checks.
        // ----------------------------------------
        const int verifyCount =
            std::min(
                3,
                int(
                    proposals.size()));

        const float inv =
            1.0f /
            std::max(
                scale,
                1e-6f);


        for(int i=0;
            i<verifyCount;
            ++i)
        {
            const auto& p =
                proposals[i];

            const cv::Point2f globalCenter(
                p.e.center.x*inv,
                p.e.center.y*inv);

            const float major =
                std::max(
                    p.e.size.width,
                    p.e.size.height) *
                inv;

            // Enough context for:
            // outer ring + H + perspective.
            const float half =
                std::max(
                    28.0f,
                    0.80f*major);

            cv::Rect2f roi(
                globalCenter.x-half,
                globalCenter.y-half,
                2.0f*half,
                2.0f*half);

            cv::Rect ir(
                clampRect(
                    roi,
                    bgr.cols,
                    bgr.rows));

            if(ir.area() <= 0)
                continue;

            cv::Mat patch =
                bgr(ir);

            const cv::Point2f localCenter(
                globalCenter.x-ir.x,
                globalCenter.y-ir.y);

            const cv::Rect2f ellipseRect =
                p.e.boundingRect2f();

            cv::Rect2f globalEllipse(
                ellipseRect.x*inv,
                ellipseRect.y*inv,
                ellipseRect.width*inv,
                ellipseRect.height*inv);

            globalEllipse =
                clampRect(
                    globalEllipse,
                    bgr.cols,
                    bgr.rows);

            cv::Rect2f localEllipse =
                globalEllipse;

            localEllipse.x -= ir.x;
            localEllipse.y -= ir.y;


            
            // ============================================================
            // V15.38 FAST RING-H GEOMETRY
            //
            // The ellipse pair is only a proposal.  H identity is mandatory.
            // We verify H directly inside the ring ROI instead of recursively
            // running the expensive RingHelper on the same patch.
            //
            // PlainH/FarH/ObliqueH remain completely independent lanes.
            // ============================================================
            std::optional<Candidate> h=
                hDetector_.findH(
                    patch,
                    localEllipse,
                    localCenter,
                    localEllipse,
                    HMode::Confirm);

            if(!h && cfg_.h_oblique_enabled){
                h=hDetector_.findOblique(
                    patch,
                    localCenter,
                    localEllipse);
            }

            if(!h && cfg_.universal_color_enabled){
                h=hDetector_.findColorInvariant(
                    patch,
                    localEllipse,
                    localCenter,
                    localEllipse,
                    HMode::Confirm);
            }

            if(!h)
                continue;

            const float ringMinor=
                std::min(p.e.size.width,p.e.size.height)*inv;
            const float ringMajor=
                std::max(p.e.size.width,p.e.size.height)*inv;
            const float hSpan=std::max(1.0f,rectSpan(h->rawBox));
            const float centerOffset=
                cv::norm(h->center-localCenter);

            // Broad projective geometry guards.  They only reject impossible
            // ring/H relationships; marginal cases remain available to the
            // unchanged DEEP_RING fallback.
            if(centerOffset>std::max(14.0f,0.65f*ringMinor))
                continue;
            if(hSpan<4.0f || hSpan>1.15f*ringMajor)
                continue;

            Candidate ring=*h;

            // Map H from patch coordinates to source coordinates.
            const cv::Point2f offset(float(ir.x),float(ir.y));
            ring.center+=offset;
            ring.box.x+=offset.x;
            ring.box.y+=offset.y;
            ring.rawBox.x+=offset.x;
            ring.rawBox.y+=offset.y;

            ring.source=CandidateSource::RingH;
            ring.hasRing=true;
            ring.ring=cv::RotatedRect(
                globalCenter,
                cv::Size2f(p.e.size.width*inv,p.e.size.height*inv),
                p.e.angle);
            ring.ringCenter=globalCenter;

            // Preserve the actual projected ellipse bounds as ring support.
            // RingHelper will expand this ROI itself during strict confirmation.
            ring.ringBox=globalEllipse;

            ring.ringPairScore=p.pairScore;
            ring.ringFitScore=p.fitScore;
            ring.shapeScore=h->score;
            ring.score=std::clamp(
                0.78f*h->score+
                0.14f*p.pairScore+
                0.08f*p.fitScore,
                0.0f,1.0f);


            return ring;
        }

        return std::nullopt;
    };


    if(state_==TrackState::Searching){

        // ================================================================
        // V15.42 H-FIRST SEARCH
        //
        // H identity owns acquisition priority. FAR / Plain / Oblique are
        // observation methods for the SAME target. Ring work is a sidecar and
        // may run only after the due H-family lane had its opportunity.
        // ================================================================

        if(searchStartFrame_<=0) searchStartFrame_=frame_;
        if(acquisitionStartFrame_<=0) acquisitionStartFrame_=frame_;

        const int acqAge=frame_-acquisitionStartFrame_;
        const bool farDue=
            cfg_.h_far_enabled &&
            !cfg_.h_far_scale_sides.empty() &&
            frame_-lastFarHFrame_>=cfg_.h_far_search_interval;
        const bool directDue=
            frame_-lastGlobalHFrame_>=cfg_.h_global_search_interval ||
            (cfg_.h_oblique_enabled &&
             frame_-lastObliqueHFrame_>=cfg_.h_oblique_search_interval);

        bool seeded=false;
        bool farTried=false;
        bool directTried=false;
        bool hLaneUsed=false;

        auto tryFarSearch=[&]() -> bool {
            if(!farDue || farTried || hLaneUsed) return false;
            farTried=true;
            hLaneUsed=true;
            detectorRan=true;
            const int side=cfg_.h_far_scale_sides[
                farScaleCursor_++%cfg_.h_far_scale_sides.size()];
            auto far=hDetector_.findFar(
                bgr,std::nullopt,std::nullopt,side);
            lastFarHFrame_=frame_;
            if(!far) return false;
            seedCandidate(*far,gray,false,CandidateSource::FarH);
            seeded=true;
            return true;
        };

        auto tryDirectSearch=[&]() -> bool {
            if(!directDue || directTried || hLaneUsed) return false;
            directTried=true;
            hLaneUsed=true;
            auto direct=runDirectGlobal();
            if(!direct) return false;
            const CandidateSource src=
                direct->source==CandidateSource::ObliqueH
                    ? CandidateSource::ObliqueH
                    : CandidateSource::PlainH;
            seedCandidate(*direct,gray,src==CandidateSource::PlainH,src);
            seeded=true;
            return true;
        };

        // V15.42 ONE H HEAVY LANE
        // Every processed SEARCH frame is useful to H identity, but we never
        // stack FAR + full-frame Plain/Oblique on the same frame. That avoids
        // the low-PROC-FPS feedback loop that previously made long-range search
        // slower in wall-clock time. During initial far grace, FAR gets 2/3 of
        // H slots; afterwards FAR and direct H alternate 1:1.
        const bool preferFar=
            cfg_.h_far_enabled &&
            (acqAge<cfg_.h_far_grace_frames
                ? (frame_%3)!=2
                : (frame_%2)==0);

        if(preferFar){
            tryFarSearch();
            if(!hLaneUsed) tryDirectSearch();
        }else{
            tryDirectSearch();
            if(!hLaneUsed) tryFarSearch();
        }

        // Fast Ring is opportunistic. It can provide an H observation/ROI but
        // it never steals a frame from a due H-family search.
        if(!seeded &&
           state_==TrackState::Searching &&
           cfg_.universal_ring_enabled &&
           (frame_%6)==0)
        {
            detectorRan=true;
            auto ringFast=runFastRingLane();
            if(ringFast){
                seedCandidate(*ringFast,gray,false,CandidateSource::RingH);
                seeded=true;
            }
        }

        // DEEP_RING is escalation only. No startup deep-ring preemption is
        // allowed in V15.42 because it was starving FAR/Plain acquisition.
        const bool deepRingDue=
            !seeded &&
            cfg_.universal_ring_enabled &&
            !ringFuture_.valid() &&
            acqAge>=36 &&
            frame_-lastRingHFrame_>=cfg_.universal_ring_interval &&
            (frame_%72)==0;

        if(deepRingDue && state_==TrackState::Searching){
            detectorRan=true;
            auto ring=ringHelper_.findRingH(
                bgr,std::nullopt,std::nullopt,std::nullopt,HMode::Acquire);
            lastRingHFrame_=frame_;
            if(ring && ring->hasRing){
                seedCandidate(*ring,gray,false,CandidateSource::RingH);
                seeded=true;
            }
        }

        // V15.43 AI IS A PARALLEL ACQUISITION SOURCE.
        // The H-family lane above is intentionally unchanged. If that lane did
        // not already acquire this frame, a completed GDINO->SAM2 batch may
        // nominate an ROI, but only its fresh current-frame H-PROOF can seed the
        // unified H identity tracklet. AI inference itself never blocks here.
        if(!seeded && state_==TrackState::Searching && aiCurrentProof){
            detectorRan=true;
            seedCandidate(
                *aiCurrentProof,gray,false,CandidateSource::AiVerifiedH);
            seeded=true;
        }
    }

    else if(state_==TrackState::Confirming){

        // ================================================================
        // V15.42 UNIFIED H CONFIRMATION
        //
        // FAR / Plain / Oblique / RingROI are detector methods, not separate
        // target identities. Every accepted current-frame H observation enters
        // one detector-to-detector tracklet. Ring evidence is classification
        // only and can never gate or reset H confirmation.
        // ================================================================

        float flowQ=0.0f;
        bool flowUpdated=false;

        // Optical flow is allowed to predict motion between detector hits, but
        // it never increments hConfirmTracklet_ and never becomes H evidence.
        if(!prevGray_.empty() && flow_.ready()){
            cv::Point2f d;
            if(flow_.update(prevGray_,gray,target_.box,velocity_,d,flowQ) &&
               flowQ>=0.10f)
            {
                const cv::Rect2f moved=clampRect(
                    moveRect(target_.box,d.x,d.y),bgr.cols,bgr.rows);
                const cv::Point2f movedCenter=rectCenter(moved);
                if(correctionAllowed(movedCenter,2.0f)){
                    velocity_=(1.0f-cfg_.velocity_alpha)*velocity_+
                              cfg_.velocity_alpha*d;
                    target_.box=moved;
                    target_.center=movedCenter;
                    cumulativeFlow_+=d;
                    if(hasRingSupport_){
                        ringSupportBox_=clampRect(
                            moveRect(ringSupportBox_,d.x,d.y),bgr.cols,bgr.rows);
                        ringSupportCenter_+=d;
                    }
                    flowUpdated=true;
                }
            }
        }

        result.flowQuality=flowQ;

        const double nowMs=
            std::chrono::duration<double,std::milli>(
                now.time_since_epoch()).count();

        auto observeRingClass=[&](const Candidate& ring) {
            if(!ring.hasRing || ring.ringPairScore<cfg_.ring_fast_min_pair)
                return;

            const float currentSpan=std::max(6.0f,rectSpan(target_.box));
            const float d=cv::norm(ring.center-target_.center);
            const float allowed=std::max(34.0f,4.0f*currentSpan);
            if(d>allowed) return;

            bool continueRing=false;
            if(haveRingClassCandidate_){
                const double gapMs=
                    std::chrono::duration<double,std::milli>(
                        now-lastRingClassObservationTime_).count();
                const float oldSpan=std::max(
                    6.0f,rectSpan(lastRingClassCandidate_.ringBox));
                const float newSpan=std::max(6.0f,rectSpan(ring.ringBox));
                const float ratio=newSpan/oldSpan;
                const float ringMove=cv::norm(
                    ring.ringCenter-lastRingClassCandidate_.ringCenter);
                const float ringAllowed=std::max(
                    38.0f,4.5f*std::max(oldSpan,newSpan));
                continueRing=
                    gapMs>=cfg_.ring_fast_min_gap_ms &&
                    gapMs<=cfg_.ring_fast_max_gap_ms &&
                    ratio>=cfg_.ring_fast_scale_ratio_min &&
                    ratio<=cfg_.ring_fast_scale_ratio_max &&
                    ringMove<=ringAllowed;
            }

            if(!haveRingClassCandidate_ || !continueRing)
                ringClassHits_=1;
            else
                ++ringClassHits_;

            lastRingClassCandidate_=ring;
            lastRingClassObservationTime_=now;
            haveRingClassCandidate_=true;
            rememberRingSupport(ring);
            fastRingHits_=ringClassHits_;

            if(ringClassHits_>=cfg_.ring_fast_required_hits){
                targetClass_=TargetClass::RingH;
                markRingProof(ring,now);
            }
        };

        auto observeH=[&](const Candidate& detected,
                          HAcquisitionMethod method,
                          bool carriesRing) -> bool {
            Candidate c=detected;

            const int oldHits=hConfirmTracklet_.hits();
            HObservation o;
            o.x=c.center.x;
            o.y=c.center.y;
            o.span=std::max(1.0f,c.span>0.0f?c.span:rectSpan(c.box));
            o.score=c.score;
            o.topology=c.topology;
            o.timeMs=nowMs;
            o.method=method;
            const bool continued=hConfirmTracklet_.observe(o);

            // A mismatched detector observation starts a NEW tentative H
            // tracklet. It does not inherit authority from the old candidate.
            if(!continued && oldHits>0){
                hConfirmStrongContextHits_=0;
                farConfirmHits_=0;
                obliqueConfirmHits_=0;
                globalConfirmHits_=0;
                resetFastObliqueProof();
            }

            acquisitionMethod_=method;
            if(method!=HAcquisitionMethod::RingRoi)
                lastHSource_=canonicalHSource(method);
            if(lastHSource_==CandidateSource::None ||
               lastHSource_==CandidateSource::RingH)
                lastHSource_=CandidateSource::PlainH;

            if(carriesRing) observeRingClass(c);

            // Current-frame detector evidence owns the confirmation position.
            // Preserve ring metadata separately when RingH classification is
            // already proven.
            const cv::Point2f oldCenter=target_.center;
            target_=c;
            target_.source=lastHSource_;
            target_.hasRing=targetClass_==TargetClass::RingH;
            if(target_.hasRing && haveRingClassCandidate_){
                target_.ring=lastRingClassCandidate_.ring;
                target_.ringBox=lastRingClassCandidate_.ringBox;
                target_.ringCenter=lastRingClassCandidate_.ringCenter;
                target_.ringPairScore=lastRingClassCandidate_.ringPairScore;
                target_.ringFitScore=lastRingClassCandidate_.ringFitScore;
            }

            const cv::Point2f detectorDelta=target_.center-oldCenter;
            velocity_=0.55f*velocity_+0.45f*detectorDelta;
            flow_.seed(gray,target_.box);

            currentHScore_=c.score;
            currentConfirmContrast_=robustCandidateContrast(gray,c);
            currentConfirmAuthority_=competitionAuthority(gray,c);

            const bool strongContext=
                currentConfirmContrast_>=82.0f ||
                (currentConfirmContrast_>=58.0f &&
                 c.score>=0.88f &&
                 c.topology>=0.90f &&
                 c.railAvg>=0.50f &&
                 c.crossbar>=0.40f &&
                 std::min(c.voidTop,c.voidBottom)>=0.55f);
            if(strongContext) ++hConfirmStrongContextHits_;

            confirmHits_=hConfirmTracklet_.hits();
            globalConfirmHits_=std::min(
                cfg_.h_global_confirm_hits,confirmHits_);
            if(method==HAcquisitionMethod::Far) ++farConfirmHits_;
            if(method==HAcquisitionMethod::Oblique) ++obliqueConfirmHits_;
            hConfirmMissesUnified_=0;
            confirmMisses_=0;
            lastHConfirmObservationTime_=now;

            if(method==HAcquisitionMethod::Oblique)
                updateFastObliqueProof(gray,c,now);

            return true;
        };

        bool hObserved=false;

        // A completed AI batch contributes only through the fresh current-frame
        // H proof produced by pollAiEvidence(). It joins the SAME tracklet and
        // cannot reset FAR/Plain/Oblique identity on its own.
        if(aiCurrentProof &&
           correctionAllowed(aiCurrentProof->center,2.2f)){
            detectorRan=true;
            hObserved=observeH(
                *aiCurrentProof,HAcquisitionMethod::Ai,false);
        }

        // V15.42 FAR-TO-PLAIN HANDOFF
        // FAR owns the first confirmation opportunity only while the H is truly
        // tiny. Once it grows, strict/local PlainH becomes canonical and FAR is
        // merely a periodic fallback. A detector-method transition still feeds
        // the SAME hConfirmTracklet_ and never resets identity.
        const bool farAvailable=
            cfg_.h_far_enabled && !cfg_.h_far_scale_sides.empty();
        const bool farPreferred=
            farAvailable &&
            ((acquisitionMethod_==HAcquisitionMethod::Far && target_.span<40.0f) ||
             target_.span<24.0f);

        // V15.45 T6 ONE-OBSERVATION AUTHORITY + LOWER-BAND CORROBORATION
        // Exactly one detector observation may advance the unified H tracklet
        // per frame.  If AI already produced current-frame H proof, FAR waits
        // until the next frame instead of observing with the same timestamp.
        if(!hObserved && farPreferred){
            detectorRan=true;
            const int side=cfg_.h_far_scale_sides[
                farScaleCursor_++%cfg_.h_far_scale_sides.size()];
            auto f=hDetector_.findFar(bgr,target_.center,target_.box,side);
            lastFarHFrame_=frame_;
            if(f){
                const bool lowerBandFar=
                    f->center.y >= 0.78f*float(bgr.rows);

                // V15.48 FAR-TO-PLAIN EARLY STRICT HANDOFF
                // Once the SAME FAR candidate is large enough for strict local
                // Plain-H geometry, try Plain before recording another FAR hit.
                // A successful strict proof advances the existing unified H
                // tracklet as PLAIN in this frame, so the normal hReady gate can
                // lock immediately when its existing hit/authority requirements
                // are satisfied.  If Plain misses, FAR remains the observation.
                //
                // T6 lower-band safety is preserved: lower-band FAR always asks
                // for strict Plain corroboration regardless of candidate size.
                const bool handoffPlainCapable=
                    std::max(f->span,target_.span)>=16.0f;

                // T6 LOWER-BAND FAR -> STRICT PLAIN CORROBORATION
                // FAR may acquire/track a tiny H in the newly extended lower
                // corridor, but runway digits/stripes cannot become green from
                // broad FAR evidence alone.  The same strict proof now also
                // performs the FAR->Plain handoff for a Plain-capable H.
                if(lowerBandFar || handoffPlainCapable){
                    const cv::Rect2f focus=expandRect(
                        f->box,1.8f,bgr.cols,bgr.rows);
                    auto strict=hDetector_.findH(
                        bgr,focus,f->center,f->box,HMode::Confirm);
                    if(strict){
                        const float d=cv::norm(strict->center-f->center);
                        const float r=std::max(
                            24.0f,1.25f*std::max(6.0f,rectSpan(f->box)));
                        if(d<=r){
                            // V15.48 HANDOFF GEOMETRY STABILIZATION
                            // The strict Plain proof owns identity/metrics, but
                            // blend its box with the already-associated FAR box
                            // to avoid a one-frame geometry jump at method handoff.
                            Candidate handoff=*strict;
                            handoff.box=blendRect(f->box,strict->box,0.25f,0.75f,
                                                  bgr.cols,bgr.rows);
                            handoff.center=rectCenter(handoff.box);
                            hObserved=observeH(
                                handoff,HAcquisitionMethod::Plain,false);
                        }
                    }
                }

                if(!hObserved)
                    hObserved=observeH(*f,HAcquisitionMethod::Far,false);
            }
        }

        // Plain H is the canonical identity verifier once enough pixels exist.
        // This is a local confirmation ROI, so it may safely follow a failed FAR
        // pass without stacking two full-frame heavy searches.
        if(!hObserved){
            detectorRan=true;
            auto h=hDetector_.findH(
                bgr,target_.box,target_.center,target_.box,HMode::Confirm);
            if(h) hObserved=observeH(*h,HAcquisitionMethod::Plain,false);
        }

        // Perspective fallback remains additive and shares the SAME H tracklet.
        //
        // V15.43 TOP-EDGE PERMISSIVE GUARD: LOCAL OBLIQUE
        const bool topEdgeLargeTentative =
            target_.span > 30.0f &&
            target_.center.y <
                std::max(
                    64.0f,
                    0.18f * float(bgr.rows));

        if(!hObserved &&
           !topEdgeLargeTentative &&
           cfg_.h_oblique_enabled &&
           target_.span >= 16.0f)
        {
            detectorRan=true;

            auto h =
                hDetector_.findOblique(
                    bgr,
                    target_.center,
                    target_.box);

            lastObliqueHFrame_=frame_;

            if(h)
                hObserved =
                    observeH(
                        *h,
                        HAcquisitionMethod::Oblique,
                        false);
        }

        // When Plain/Oblique misses after the H has grown, FAR is still allowed
        // to rescue the same tracklet occasionally, but it no longer monopolizes
        // every confirmation frame or keeps a large near-field H labelled FAR.
        if(!hObserved && !farPreferred && farAvailable &&
           target_.span<=135.0f && (frame_%3)==0){
            detectorRan=true;
            const int side=cfg_.h_far_scale_sides[
                farScaleCursor_++%cfg_.h_far_scale_sides.size()];
            auto f=hDetector_.findFar(bgr,target_.center,target_.box,side);
            lastFarHFrame_=frame_;
            if(f) hObserved=observeH(*f,HAcquisitionMethod::Far,false);
        }

        // If the local prediction is wrong, a periodic full-frame H family
        // search can replace the tentative tracklet. Ring never monopolizes it.
        if(!hObserved && (frame_%4)==1){
            auto direct=runDirectGlobal();
            if(direct){
                const auto method=methodFromSource(direct->source);
                hObserved=observeH(*direct,method,false);
            }
        }

        // V15.42 RING RESCUE ONLY WHEN H MISSED
        // A successful FAR/Plain/Oblique observation is enough to advance H
        // confirmation immediately. Ring work is not allowed to add latency to
        // that frame. Fast Ring is consulted only as an H+ring rescue when the
        // H-family confirmation paths all missed; classification can catch up
        // after H is LOCKED.
        if(!hObserved && cfg_.universal_ring_enabled && (frame_%4)==0){
            detectorRan=true;
            auto ringFast=runFastRingLane();
            if(ringFast){
                const float d=cv::norm(ringFast->center-target_.center);
                const float allowed=std::max(
                    38.0f,4.5f*std::max(6.0f,rectSpan(target_.box)));
                if(d<=allowed){
                    hObserved=observeH(
                        *ringFast,HAcquisitionMethod::RingRoi,true);
                }
            }
        }

        if(!hObserved){
            ++hConfirmMissesUnified_;
            ++confirmMisses_;
            currentHScore_*=0.98f;
        }

        confirmHits_=hConfirmTracklet_.hits();
        globalConfirmHits_=std::min(cfg_.h_global_confirm_hits,confirmHits_);
        fastRingHits_=ringClassHits_;
        ringIndependentHHits_=confirmHits_;

        const int requiredHits=
            (hConfirmTracklet_.lastMethod()==HAcquisitionMethod::Far &&
             target_.span<40.0f)
                ? cfg_.h_far_required_hits
                : cfg_.h_confirm_hits;

        const double sinceLastHMs=
            lastHConfirmObservationTime_.time_since_epoch().count()==0
                ? 1.0e9
                : std::chrono::duration<double,std::milli>(
                    now-lastHConfirmObservationTime_).count();

        // High-contrast/strong-geometry H locks at the normal three-hit rule.
        // Low-contrast H is NOT rejected; it simply needs two extra repeated
        // current-frame detector observations before becoming green authority.
        const bool lockAuthority=
            hConfirmStrongContextHits_>0 ||
            hConfirmTracklet_.hits()>=requiredHits+2;
        // V15.45 T6 LOWER-BAND STRICT AUTHORITY
        const HAcquisitionMethod lastConfirmMethod=
            hConfirmTracklet_.lastMethod();
        const bool lowerBandNeedsStrictProof=
            target_.center.y >= 0.78f*float(bgr.rows) &&
            (lastConfirmMethod==HAcquisitionMethod::Far ||
             lastConfirmMethod==HAcquisitionMethod::Oblique ||
             lastConfirmMethod==HAcquisitionMethod::Ai);

        // V15.53 SUPERFAST LARGE PLAIN AUTHORITY
        // A large H that is already returned by the existing strict Plain-H
        // detector no longer waits for the low-context +2-hit fallback.
        // Never lock from a single frame: require one prior temporally/spatially
        // consistent H observation plus a fresh CURRENT-FRAME strict Plain proof.
        // FAR/Oblique/AI/flow cannot enter this fast path, and the lower runway
        // band stays excluded. Detector thresholds are unchanged.
        const bool largePlainFastAuthority=
            hObserved &&
            sinceLastHMs<=260.0 &&
            lastConfirmMethod==HAcquisitionMethod::Plain &&
            hConfirmTracklet_.hits()>=2 &&
            target_.span>=30.0f &&
            target_.center.y < 0.78f*float(bgr.rows) &&
            strongIdentityProof(target_);

        const bool hReady=
            largePlainFastAuthority ||
            (hConfirmTracklet_.ready(requiredHits) &&
             sinceLastHMs<=260.0 &&
             lockAuthority &&
             !lowerBandNeedsStrictProof);
        if(hReady){
            state_=TrackState::Locked;
            lostFrames_=0;
            identityMisses_=0;
            confirmationFailures_=0;
            acquisitionStartFrame_=0;
            flow_.seed(gray,target_.box);

            if(targetClass_==TargetClass::RingH &&
               haveRingClassCandidate_)
                markRingProof(lastRingClassCandidate_,
                              lastRingClassObservationTime_);

            // Current-frame detector observation becomes the first immutable
            // identity proof. Flow played no role in this transition.
            markIdentityProof(target_,now);
        }else{
            const bool staleTracklet=sinceLastHMs>950.0;
            const bool tooManyMisses=
                hConfirmMissesUnified_>=6 && sinceLastHMs>320.0;

            if(staleTracklet || tooManyMisses){
                ++confirmationFailures_;
                state_=TrackState::Searching;
                clearTarget();
                searchStartFrame_=frame_;
                lastGlobalHFrame_=frame_-cfg_.h_global_search_interval;
                lastFarHFrame_=frame_-cfg_.h_far_search_interval;
                lastObliqueHFrame_=frame_-cfg_.h_oblique_search_interval;
            }
        }
    }

    else if(state_==TrackState::Locked ||
            state_==TrackState::Suspect){

        const bool wasLockedAtFrameStart=
            state_==TrackState::Locked;

        // ================================================================
        // V15.38 NON-WEAKENING LOCK
        //
        // Optical flow owns MOTION only.
        // Fresh H evidence owns IDENTITY only.
        // Ring geometry is additional classification/support only.
        //
        // No global detector may steal a healthy target in this state.
        // ================================================================

        float flowQ=0.0f;
        bool flowUpdated=false;

        if(!prevGray_.empty() && flow_.ready()){
            cv::Point2f d;

            if(flow_.update(
                   prevGray_,
                   gray,
                   target_.box,
                   velocity_,
                   d,
                   flowQ) &&
               flowQ>=0.10f)
            {
                velocity_=
                    (1.0f-cfg_.velocity_alpha)*velocity_+
                    cfg_.velocity_alpha*d;

                const cv::Rect2f pb=
                    clampRect(
                        moveRect(target_.box,d.x,d.y),
                        bgr.cols,bgr.rows);

                const cv::Point2f pc=rectCenter(pb);

                if(correctionAllowed(pc,1.7f)){
                    target_.box=pb;
                    target_.center=
                        (1.0f-cfg_.center_alpha)*target_.center+
                        cfg_.center_alpha*pc;

                    cumulativeFlow_+=d;

                    if(hasRingSupport_){
                        ringSupportBox_=
                            clampRect(
                                moveRect(ringSupportBox_,d.x,d.y),
                                bgr.cols,bgr.rows);
                        ringSupportCenter_+=d;
                    }

                    flowUpdated=true;
                    lostFrames_=0;
                }
            }
        }

        if(!flowUpdated)
            ++lostFrames_;

        const bool healthyFlow=
            flowUpdated && flowQ>=0.18f;

        // V15.49 STRONG-FLOW BOUNDED IDENTITY GRACE
        // Preserve an already-proven LOCK through a short verifier gap only
        // when current optical motion is exceptionally strong and has already
        // passed correctionAllowed() spatial association above. Flow does NOT
        // create or renew identity: identityAge keeps increasing, strict H
        // verification still runs on the original T7/urgent cadence, and the
        // hard identity expiry remains unchanged. A SUSPECT target cannot use
        // this grace to promote itself back to LOCKED without fresh H proof.
        const bool strongOwnedFlowGrace=
            wasLockedAtFrameStart &&
            haveIdentityProof_ &&
            flowUpdated &&
            flowQ>=0.95f;
        if(strongOwnedFlowGrace){
            const float graceCapMs=std::min(
                460.0f,
                static_cast<float>(cfg_.identity_hard_lease_ms)-1.0f);
            if(graceCapMs>effectiveIdentitySoftLeaseMs)
                effectiveIdentitySoftLeaseMs=graceCapMs;
        }

        // V15.46 CURRENT-FRAME LOCK KEEPER
        //
        // T7: a correct H lock was expiring because async identity proof did
        // not return before the 320 ms soft lease under throttled processing.
        // Run ONLY the existing strict local Plain-H tracker on the current
        // target ROI once half of the identity lease is consumed.
        //
        // A successful current-frame detector proof renews identity only.
        // It never moves the target; optical flow retains motion ownership.
        // Expensive Oblique/FAR/Color recovery remains asynchronous.
        bool currentFrameLockProof=false;
        const float localProofAge=
            haveIdentityProof_ ? identityAgeMs(now) : 1.0e9f;

        if(localProofAge>=0.50f*cfg_.identity_soft_lease_ms &&
           rectSpan(target_.box)>=12.0f)
        {
            detectorRan=true;
            auto localLockProof=hDetector_.findH(
                bgr,
                target_.box,
                target_.center,
                target_.box,
                HMode::Track);

            if(localLockProof &&
               strongIdentityProof(*localLockProof) &&
               correctionAllowed(localLockProof->center,1.35f))
            {
                currentFrameLockProof=true;
                currentHScore_=
                    0.70f*currentHScore_+0.30f*localLockProof->score;
                acquisitionMethod_=HAcquisitionMethod::Plain;
                lastHSource_=CandidateSource::PlainH;
                target_.source=CandidateSource::PlainH;

                // V15.46 proof renews identity only; optical flow keeps motion ownership.
                markIdentityProof(*localLockProof,now);
                verifiedFlowAccum_=cumulativeFlow_;
                identityMisses_=0;
                lostFrames_=0;
                state_=TrackState::Locked;
            }
        }

        // AI evidence itself never refreshes identity. This candidate is a
        // detector-backed proof made synchronously on THIS frame inside the
        // bounded AI ROI, so it may renew identity just like any other fresh H
        // verification after spatial association succeeds.
        if(aiCurrentProof &&
           strongIdentityProof(*aiCurrentProof) &&
           correctionAllowed(aiCurrentProof->center,1.8f)){
            detectorRan=true;
            const Candidate proof=*aiCurrentProof;
            const cv::Point2f oldCenter=target_.center;
            target_.box=blendRect(
                target_.box,proof.box,0.35f,0.65f,bgr.cols,bgr.rows);
            target_.center=rectCenter(target_.box);
            target_.source=CandidateSource::AiVerifiedH;
            acquisitionMethod_=HAcquisitionMethod::Ai;
            lastHSource_=CandidateSource::AiVerifiedH;
            velocity_=0.70f*velocity_+0.30f*(target_.center-oldCenter);
            currentHScore_=proof.score;
            markIdentityProof(proof,now);
            verifiedFlowAccum_=cumulativeFlow_;
            flow_.seed(gray,target_.box);
            state_=TrackState::Locked;
            identityMisses_=0;
            lostFrames_=0;
        }

        // Consume completed background verification without blocking this
        // frame.  The workers themselves only ever keep one latest request.
        const bool hResultConsumed=
            pollIdentityVerification(gray,now);
        const bool ringResultConsumed=
            pollRingVerification(gray,now);

        if(hResultConsumed || ringResultConsumed)
            detectorRan=true;

        auto elapsedSince =
            [&](Clock::time_point t) -> float {
                if(t.time_since_epoch().count()==0) return 1.0e9f;
                return static_cast<float>(
                    std::chrono::duration<double,std::milli>(now-t).count());
            };

        float idAge=identityAgeMs(now);

        // A suspect/stale identity gets a faster local verifier cadence, but
        // the worker is still single-flight so this can never create a queue.
        const bool urgentIdentity=
            state_==TrackState::Suspect ||
            idAge>0.70f*cfg_.identity_soft_lease_ms ||
            !healthyFlow;

        const int verifyPeriodMs=
            urgentIdentity
                ? cfg_.locked_identity_urgent_period_ms
                : cfg_.locked_identity_period_ms;

        if(!currentFrameLockProof &&
           elapsedSince(lastIdentityRequestTime_)>=verifyPeriodMs){
            queueIdentityVerification(
                bgr,
                now,
                state_==TrackState::Suspect ||
                idAge>cfg_.identity_soft_lease_ms);
            detectorRan=true;
        }

        // ================================================================
        // V15.42 LOCKED FAST RING SIDECAR
        //
        // Plain/FAR/Oblique H never pays for a continuous DEEP_RING worker.
        // A cheap current-frame Fast Ring observation first creates ring
        // evidence. Only then may the strict background RingHelper confirm the
        // class. Neither path can renew H identity.
        // ================================================================
        if(cfg_.universal_ring_enabled &&
           rectSpan(target_.box)>=10.0f &&
           (frame_%12)==0)
        {
            detectorRan=true;
            auto ringFast=runFastRingLane();
            if(ringFast){
                const float currentSpan=std::max(6.0f,rectSpan(target_.box));
                const float d=cv::norm(ringFast->center-target_.center);
                const float allowed=std::max(34.0f,4.0f*currentSpan);

                if(d<=allowed){
                    bool continueRing=false;
                    if(haveRingClassCandidate_){
                        const double gapMs=
                            std::chrono::duration<double,std::milli>(
                                now-lastRingClassObservationTime_).count();
                        const float oldSpan=std::max(
                            6.0f,rectSpan(lastRingClassCandidate_.ringBox));
                        const float newSpan=std::max(
                            6.0f,rectSpan(ringFast->ringBox));
                        const float ratio=newSpan/oldSpan;
                        const float move=cv::norm(
                            ringFast->ringCenter-
                            lastRingClassCandidate_.ringCenter);
                        const float moveAllowed=std::max(
                            38.0f,4.5f*std::max(oldSpan,newSpan));
                        continueRing=
                            gapMs>=cfg_.ring_fast_min_gap_ms &&
                            gapMs<=cfg_.ring_fast_max_gap_ms &&
                            ratio>=cfg_.ring_fast_scale_ratio_min &&
                            ratio<=cfg_.ring_fast_scale_ratio_max &&
                            move<=moveAllowed;
                    }

                    ringClassHits_=
                        (!haveRingClassCandidate_ || !continueRing)
                            ? 1
                            : ringClassHits_+1;
                    lastRingClassCandidate_=*ringFast;
                    lastRingClassObservationTime_=now;
                    haveRingClassCandidate_=true;
                    rememberRingSupport(*ringFast);
                    fastRingHits_=ringClassHits_;

                    // Preserve only current-frame ring geometry for display.
                    target_.ring=ringFast->ring;
                    target_.ringBox=ringFast->ringBox;
                    target_.ringCenter=ringFast->ringCenter;
                    target_.ringPairScore=ringFast->ringPairScore;
                    target_.ringFitScore=ringFast->ringFitScore;

                    if(ringClassHits_>=cfg_.ring_fast_required_hits){
                        targetClass_=TargetClass::RingH;
                        target_.hasRing=true;
                        markRingProof(*ringFast,now);
                    }

                    // A current Fast Ring proof makes an immediate expensive
                    // strict ring launch unnecessary.
                    lastRingRequestTime_=now;
                }
            }
        }

        // Strict RingHelper is escalation/maintenance only after ring evidence
        // exists. A normal H lock therefore never loses CPU to blind DEEP_RING.
        const bool ringEvidenceActive=
            targetClass_==TargetClass::RingH || hasRingSupport_;
        const int ringPeriodMs=
            targetClass_==TargetClass::RingH
                ? cfg_.locked_ring_period_ms
                : std::max(1400,2*cfg_.locked_ring_period_ms);
        const bool wantRingCheck=
            cfg_.universal_ring_enabled &&
            ringEvidenceActive &&
            rectSpan(target_.box)>=10.0f &&
            elapsedSince(lastRingRequestTime_)>=ringPeriodMs;

        if(wantRingCheck){
            queueRingVerification(
                bgr,
                now,
                state_==TrackState::Suspect);
            detectorRan=true;
        }

        idAge=identityAgeMs(now);

        // Ring label is a classification lease, not an H identity lease.
        if(targetClass_==TargetClass::RingH &&
           ringAgeMs(now)>cfg_.ring_label_lease_ms &&
           haveIdentityProof_)
        {
            targetClass_=TargetClass::PlainH;
            target_.hasRing=false;
            ringClassHits_=0;
            haveRingClassCandidate_=false;
            hasRingSupport_=false;
            haveRingProof_=false;
        }

        // ---------------------------------------------------------------
        // Identity state machine.
        // ---------------------------------------------------------------
        if(idAge<=effectiveIdentitySoftLeaseMs){
            state_=TrackState::Locked;
        }else{
            state_=TrackState::Suspect;
            currentHScore_*=0.995f;
        }

        // FLOW=1.0 can no longer preserve a dead identity forever.
        // Hard identity expiry is sufficient to leave the owned target even
        // when LK is happily following grass/runway texture.
        const bool hardIdentityExpired=
            idAge>cfg_.identity_hard_lease_ms;

        const bool motionAndIdentityBad=
            lostFrames_>cfg_.max_lost_frames &&
            idAge>cfg_.identity_soft_lease_ms;

        if(hardIdentityExpired || motionAndIdentityBad){
            state_=TrackState::Searching;
            clearTarget();
            searchStartFrame_=frame_;
            acquisitionStartFrame_=frame_;
            confirmationFailures_=0;
            lastGlobalHFrame_=frame_-cfg_.h_global_search_interval;
            lastFarHFrame_=frame_-cfg_.h_far_search_interval;
            lastObliqueHFrame_=frame_-cfg_.h_oblique_search_interval;
        }

        // Healthy owned targets never accept a global takeover.  Reacquisition
        // happens only after identity authority expires and SEARCHING resumes.
        if(state_!=TrackState::Searching){
            pendingTakeover_.reset();
            takeoverHits_=0;
            lastGlobalHFrame_=frame_;
        }

        result.flowQuality=flowQ;
    }

    // V15.43 latest-frame AI scheduling happens after the real-time perception
    // work. The copy is either the full search frame or a bounded target ROI; a
    // newer submit replaces any waiting request instead of building latency.
    submitAiRequest(bgr,now);

    prevGray_=std::move(gray);
    if(target_.source!=CandidateSource::None)target_.score=currentHScore_;

    // V15.42 separates detector origin from target classification.  Internal
    // target_.source remains H-method metadata; the public/UI copy is upgraded
    // to RING_H only while a ring classification lease is fresh.
    Candidate outputTarget=target_;
    const bool ringClassFresh=
        targetClass_==TargetClass::RingH &&
        haveRingProof_ &&
        ringAgeMs(now)<=cfg_.ring_label_lease_ms;
    if(outputTarget.source!=CandidateSource::None){
        outputTarget.source=ringClassFresh
            ? CandidateSource::RingH
            : (lastHSource_==CandidateSource::None ||
               lastHSource_==CandidateSource::RingH
                   ? CandidateSource::PlainH
                   : lastHSource_);
        if(!ringClassFresh) outputTarget.hasRing=false;
    }

    result.state=state_;
    result.identityAgeMs=haveIdentityProof_ ? identityAgeMs(now) : -1.0f;
    result.ringAgeMs=haveRingProof_ ? ringAgeMs(now) : -1.0f;
    result.identityValid=
        state_==TrackState::Locked &&
        haveIdentityProof_ &&
        result.identityAgeMs<=effectiveIdentitySoftLeaseMs;

    result.guidanceValid=result.identityValid;

    // Only a fresh LOCK is an authoritative target.  CONFIRMING and SUSPECT
    // remain visible as non-authoritative predictions/candidates, so UI and
    // diagnostics keep all information without exposing guidance authority.
    result.hasTarget=
        target_.source!=CandidateSource::None &&
        result.identityValid;

    result.hasPrediction=
        target_.source!=CandidateSource::None &&
        (state_==TrackState::Confirming ||
         state_==TrackState::Suspect);

    if(result.hasTarget || result.hasPrediction){
        result.target=outputTarget;
    }

    if(result.guidanceValid){
        result.guidance=calculateGuidance(target_.center,bgr.cols,bgr.rows,cfg_);
    }

    result.identityMisses=identityMisses_;
    result.flowLostFrames=lostFrames_;
    result.confirmHits=confirmHits_;
    result.globalConfirmHits=globalConfirmHits_;
    result.farConfirmHits=farConfirmHits_;
    result.obliqueConfirmHits=obliqueConfirmHits_;
    result.confirmationFailures=confirmationFailures_;
    result.confirmAuthority=currentConfirmAuthority_;
    result.confirmContrast=currentConfirmContrast_;
    result.fastObliqueHits=fastObliqueHits_;
    result.fastRingHits=fastRingHits_;
    result.ringIndependentHHits=ringIndependentHHits_;
    result.detectorRan=detectorRan;

    // V15.43 telemetry only. None of these fields participates in H authority.
    result.gdinoOn=aiEnabled_.load(std::memory_order_acquire) && aiDinoReady_;
    result.sam2On=aiEnabled_.load(std::memory_order_acquire) && aiSamReady_;
    result.aiFps=aiFps_.load(std::memory_order_relaxed);
    result.aiResultAgeMs=lastAiEvidenceCaptureTime_.time_since_epoch().count()==0
        ? -1.0f
        : static_cast<float>(std::chrono::duration<double,std::milli>(
              now-lastAiEvidenceCaptureTime_).count());
    result.gdinoScore=lastDinoScore_;
    result.samScore=lastSamScore_;
    result.aiQueueReplaced=static_cast<int>(std::min<std::size_t>(
        aiRequests_.replacedCount(),static_cast<std::size_t>(2147483647)));
    result.hProofScore=currentHScore_;
    result.hProofMode=target_.source==CandidateSource::None
        ? "NONE" : hProofModeName(target_);
    result.acquisitionMethod=acquisitionMethodName(acquisitionMethod_);
    result.targetClass=targetClassName(targetClass_);

    std::ostringstream ss;
    switch(state_){
        case TrackState::Searching:  ss<<"SEARCHING"; break;
        case TrackState::Confirming: ss<<"CONFIRMING"; break;
        case TrackState::Locked:     ss<<"LOCKED"; break;
        case TrackState::Suspect:    ss<<"SUSPECT"; break;
    }
    if(result.hasTarget || result.hasPrediction)
        ss<<" "<<sourceName(outputTarget.source)<<" score="<<currentHScore_;
    result.status=ss.str();
    return result;
}

} // namespace helipad
