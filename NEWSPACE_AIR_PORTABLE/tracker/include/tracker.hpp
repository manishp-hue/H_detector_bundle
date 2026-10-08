#pragma once

#include <opencv2/opencv.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "ai_latest_slot.hpp"
#include "h_identity_tracklet.hpp"
#include "trt_runtime.hpp"
#include "vision.hpp"

namespace helipad {

class CoreLock05Primary;

enum class TrackState { Searching, Confirming, Locked, Suspect };

struct TrackResult {
    TrackState state = TrackState::Searching;

    // hasTarget means the candidate is safe to expose as an authoritative
    // target.  SUSPECT predictions intentionally do NOT set this flag.
    bool hasTarget = false;

    // CONFIRMING/SUSPECT may still carry a non-authoritative candidate or
    // motion-only prediction for display/recovery.  It must never be used as
    // guidance authority.
    bool hasPrediction = false;
    bool identityValid = false;
    bool guidanceValid = false;

    Candidate target{};
    Guidance guidance{};
    float flowQuality = 0.0f;
    float identityAgeMs = 0.0f;
    float ringAgeMs = 0.0f;
    int identityMisses = 0;
    int flowLostFrames = 0;
    int confirmHits = 0;
    int globalConfirmHits = 0;
    int farConfirmHits = 0;
    int obliqueConfirmHits = 0;
    int confirmationFailures = 0;

    // V15.39 acquisition diagnostics.  These are telemetry only and never
    // become guidance authority by themselves.
    float confirmAuthority = 0.0f;
    float confirmContrast = 0.0f;
    int fastObliqueHits = 0;

    // V15.40 RingH confirmation diagnostics.
    int fastRingHits = 0;
    int ringIndependentHHits = 0;

    // Telemetry only:
    // true when H/FAR/OBLIQUE/RING/AI identity detection ran this frame.
    bool detectorRan = false;

    // V15.43 asynchronous AI observability. These values never grant target
    // authority; they only expose health/evidence from the latest AI worker.
    bool gdinoOn = false;
    bool sam2On = false;
    float aiFps = 0.0f;
    float aiResultAgeMs = -1.0f;
    float gdinoScore = 0.0f;
    float samScore = 0.0f;
    int aiQueueReplaced = 0;
    float hProofScore = 0.0f;
    std::string hProofMode = "NONE";
    std::string acquisitionMethod = "NONE";
    std::string targetClass = "PLAIN_H";

    std::string status;
};

class HelipadTracker {
public:
    explicit HelipadTracker(VisionConfig cfg);
    ~HelipadTracker();
    bool setAiEngines(const std::string& dinoPlan, const std::string& samPlan = {});
    void setAiEnabled(bool enabled);
    bool aiEnabled() const;
    TrackResult process(const cv::Mat& bgr);
    void reset();

private:
    using Clock = std::chrono::steady_clock;

    struct IdentityRequest {
        cv::Mat frame;
        Candidate targetSnapshot{};
        Candidate verifiedSnapshot{};
        bool haveVerifiedSnapshot = false;
        bool recovery = false;
        uint64_t generation = 0;
        uint64_t serial = 0;
        cv::Point2f flowAccumAtSubmit{0,0};
        Clock::time_point capturedAt{};
    };

    struct IdentityResult {
        IdentityRequest request{};
        std::optional<Candidate> h;
    };

    struct RingRequest {
        cv::Mat frame;
        Candidate targetSnapshot{};
        Candidate verifiedSnapshot{};
        bool haveVerifiedSnapshot = false;
        bool recovery = false;
        bool haveRingSupport = false;
        cv::Rect2f ringSupportBox{};
        uint64_t generation = 0;
        uint64_t serial = 0;
        cv::Point2f flowAccumAtSubmit{0,0};
        Clock::time_point capturedAt{};
    };

    struct RingResult {
        RingRequest request{};
        std::optional<Candidate> ring;
    };

    struct AiRequest {
        // `frame` is full-size while searching and a cropped current-target ROI
        // while confirming/locked. Offset restores evidence to source pixels.
        cv::Mat frame;
        cv::Point2f offset{0,0};
        cv::Size sourceSize{};
        std::optional<cv::Rect2f> samPrompt;
        uint64_t generation = 0;
        uint64_t serial = 0;
        bool recovery = false;
        Clock::time_point capturedAt{};
    };

    struct AiEvidenceItem {
        cv::Rect2f dinoBox{};
        cv::Rect2f samBox{};
        float dinoScore = 0.0f;
        float samScore = 0.0f;
        bool fromClassicalPrompt = false;
    };

    struct AiEvidence {
        uint64_t generation = 0;
        uint64_t serial = 0;
        bool recovery = false;
        Clock::time_point capturedAt{};
        Clock::time_point finishedAt{};
        std::vector<AiEvidenceItem> items;
    };

    VisionConfig cfg_;

    // V15.56 PRIMARY AUTHORITY: PureH_CoreLock_05 owns classical H
    // acquisition, confirmation, identity, tracking and bounded recovery.
    // PIMPL-style ownership keeps the large upstream header out of this public
    // interface and prevents any impact on the TensorRT AI declarations.
    std::unique_ptr<CoreLock05Primary> coreLock05_;

    HGeometryDetector hDetector_;
    // V15.42 isolates background identity and ring work from the primary H
    // detector so asynchronous sidecars never share detector state.
    HGeometryDetector identityDetector_;
    HGeometryDetector ringDetector_;
    HGeometryDetector aiProofDetector_;
    RingHelper ringHelper_;
    OpticalTracker flow_;
    TensorRtAiRecovery ai_;

    // V15.43: one dedicated TensorRT worker. `LatestSlot` holds at most one
    // not-yet-started frame, so a newer request replaces stale pending work.
    LatestSlot<AiRequest> aiRequests_;
    std::thread aiWorker_;
    std::atomic<bool> aiStop_{false};
    std::atomic<bool> aiEnabled_{false};
    std::atomic<float> aiFps_{0.0f};
    bool aiDinoReady_ = false;
    bool aiSamReady_ = false;
    uint64_t aiSerial_ = 0;
    Clock::time_point lastAiSubmitTime_{};
    mutable std::mutex aiResultMutex_;
    std::optional<AiEvidence> latestAiEvidence_;
    float lastAiResultAgeMs_ = -1.0f;
    Clock::time_point lastAiEvidenceCaptureTime_{};
    float lastDinoScore_ = 0.0f;
    float lastSamScore_ = 0.0f;

    TrackState state_ = TrackState::Searching;
    Candidate target_{};

    // V15.42: detector origin, target identity, and ring classification are
    // separate concepts. CandidateSource remains detector/UI metadata only.
    HAcquisitionMethod acquisitionMethod_ = HAcquisitionMethod::None;
    TargetClass targetClass_ = TargetClass::PlainH;
    CandidateSource lastHSource_ = CandidateSource::PlainH;
    HIdentityTracklet hConfirmTracklet_{};
    int hConfirmMissesUnified_ = 0;
    int hConfirmStrongContextHits_ = 0;
    Clock::time_point lastHConfirmObservationTime_{};

    // Ring classification sidecar. These counters never gate H identity.
    int ringClassHits_ = 0;
    Clock::time_point lastRingClassObservationTime_{};
    Candidate lastRingClassCandidate_{};
    bool haveRingClassCandidate_ = false;

    cv::Mat prevGray_;
    cv::Point2f velocity_{0, 0};

    int frame_ = 0;
    int searchStartFrame_ = 0;
    int acquisitionStartFrame_ = 0;
    int confirmStartFrame_ = -100000;
    int lastGlobalHFrame_ = -100000;
    int lastFarHFrame_ = -100000;
    int lastHVerifyFrame_ = -100000;
    int lastRingHFrame_ = -100000;
    int lastObliqueHFrame_ = -100000;
    int lastDinoFrame_ = -100000;
    int farScaleCursor_ = 0;

    int lostFrames_ = 0;
    int identityMisses_ = 0;
    int confirmHits_ = 0;
    int confirmMisses_ = 0;
    int globalConfirmHits_ = 0;
    int farConfirmHits_ = 0;
    int farConfirmMisses_ = 0;
    int obliqueConfirmHits_ = 0;
    int obliqueConfirmMisses_ = 0;
    int confirmationFailures_ = 0;
    float currentHScore_ = 0.0f;

    std::optional<Candidate> pendingTakeover_;
    int takeoverHits_ = 0;

    // Two consistent H+circle observations may replace an incorrectly owned
    // FAR/OBLIQUE/AI target during confirmation.  Healthy LOCKED targets never
    // accept a global takeover.
    std::optional<Candidate> pendingRingTakeover_;
    int ringTakeoverHits_ = 0;

    cv::Rect2f ringSupportBox_{};
    cv::Point2f ringSupportCenter_{};
    bool hasRingSupport_ = false;
    int ringSupportMisses_ = 0;

    // ---------------------------------------------------------------------
    // V15.38 identity authority.
    // ---------------------------------------------------------------------
    uint64_t targetGeneration_ = 0;
    uint64_t identitySerial_ = 0;
    uint64_t ringSerial_ = 0;

    bool haveIdentityProof_ = false;
    bool haveRingProof_ = false;
    bool haveVerifiedTarget_ = false;
    Candidate verifiedTarget_{};

    Clock::time_point lastIdentityProofTime_{};
    Clock::time_point lastRingProofTime_{};
    Clock::time_point lastIdentityRequestTime_{};
    Clock::time_point lastRingRequestTime_{};

    // V15.39 confirmation authority / candidate-competition state.
    Clock::time_point confirmStartTime_{};
    Clock::time_point lastStrongObliqueProofTime_{};
    Clock::time_point lastObliqueFastVerifyTime_{};
    Clock::time_point lastConfirmCompetitionTime_{};
    Clock::time_point lastConfirmRingAssistTime_{};
    Candidate lastStrongObliqueCandidate_{};
    bool haveStrongObliqueCandidate_ = false;
    int fastObliqueHits_ = 0;
    float currentConfirmAuthority_ = 0.0f;
    float currentConfirmContrast_ = 0.0f;
    bool confirmCompetitionUseFar_ = true;

    // V15.40 RingH temporal confirmation state.
    Clock::time_point lastStrongRingProofTime_{};
    Clock::time_point lastRingFastVerifyTime_{};
    Clock::time_point lastRingDeepConfirmTime_{};
    Candidate lastStrongRingCandidate_{};
    bool haveStrongRingCandidate_ = false;
    int fastRingHits_ = 0;
    int ringIndependentHHits_ = 0;
    CandidateSource ringIndependentSource_ = CandidateSource::PlainH;

    cv::Point2f cumulativeFlow_{0,0};
    // Flow accumulator at the instant verifiedTarget_ was positioned in the
    // current coordinate system.  This lets recovery propagate the last real
    // H proof with subsequent optical motion without pretending flow is a new
    // identity proof.
    cv::Point2f verifiedFlowAccum_{0,0};

    std::future<IdentityResult> identityFuture_;
    std::optional<IdentityRequest> pendingIdentity_;

    std::future<RingResult> ringFuture_;
    std::optional<RingRequest> pendingRing_;

    void clearTarget();
    void seedCandidate(const Candidate& c, const cv::Mat& gray,
                       bool globalSeed, CandidateSource source);
    void rememberRingSupport(const Candidate& c);
    bool correctionAllowed(const cv::Point2f& candidateCenter,
                           float factor) const;

    void invalidateIdentityAuthority();
    void markIdentityProof(const Candidate& proof, Clock::time_point proofTime);
    void markRingProof(const Candidate& ring, Clock::time_point proofTime);
    float identityAgeMs(Clock::time_point now) const;
    float ringAgeMs(Clock::time_point now) const;
    bool strongIdentityProof(const Candidate& c) const;
    bool strongRingIdentityRescue(const Candidate& c) const;
    float robustCandidateContrast(const cv::Mat& gray, const Candidate& c) const;
    float obliqueConfirmationAuthority(const cv::Mat& gray, const Candidate& c,
                                       float* contrastOut = nullptr) const;
    float competitionAuthority(const cv::Mat& gray, const Candidate& c) const;
    bool updateFastObliqueProof(const cv::Mat& gray, const Candidate& c,
                                Clock::time_point now);
    void resetFastObliqueProof();
    bool updateFastRingProof(const Candidate& c, Clock::time_point now);
    void resetFastRingProof();

    void queueIdentityVerification(const cv::Mat& bgr, Clock::time_point now,
                                   bool recovery);
    void queueRingVerification(const cv::Mat& bgr, Clock::time_point now,
                               bool recovery);
    void launchIdentityRequest(IdentityRequest request);
    void launchRingRequest(RingRequest request);
    bool pollIdentityVerification(const cv::Mat& gray, Clock::time_point now);
    bool pollRingVerification(const cv::Mat& gray, Clock::time_point now);

    TrackResult processCoreLock05(const cv::Mat& bgr, Clock::time_point now);

    void aiWorkerLoop();
    void stopAiWorker();
    void submitAiRequest(const cv::Mat& bgr, Clock::time_point now);
    std::optional<Candidate> pollAiEvidence(const cv::Mat& bgr,
                                            Clock::time_point now);
};

} // namespace helipad
