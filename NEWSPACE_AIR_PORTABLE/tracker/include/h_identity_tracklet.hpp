#pragma once

#include <algorithm>
#include <cmath>

namespace helipad {

enum class HAcquisitionMethod {
    None,
    Far,
    Plain,
    Oblique,
    RingRoi,
    Ai
};

enum class TargetClass {
    PlainH,
    RingH
};

inline const char* acquisitionMethodName(HAcquisitionMethod m) {
    switch (m) {
        case HAcquisitionMethod::Far: return "FAR";
        case HAcquisitionMethod::Plain: return "PLAIN";
        case HAcquisitionMethod::Oblique: return "OBLIQUE";
        case HAcquisitionMethod::RingRoi: return "RING_ROI";
        case HAcquisitionMethod::Ai: return "AI";
        default: return "NONE";
    }
}

struct HObservation {
    double x = 0.0;
    double y = 0.0;
    double span = 0.0;
    double score = 0.0;
    double topology = 0.0;
    double timeMs = 0.0;
    HAcquisitionMethod method = HAcquisitionMethod::None;
};

// Detector-observation continuity only.  Optical flow is deliberately absent
// from this type so flow can never become confirmation/identity evidence.
class HIdentityTracklet {
public:
    void reset() {
        have_ = false;
        hits_ = 0;
        vx_ = 0.0;
        vy_ = 0.0;
        last_ = {};
    }

    bool observe(const HObservation& o) {
        if (!have_) {
            start(o);
            return false;
        }

        const double dtMs = o.timeMs - last_.timeMs;
        const double oldSpan = std::max(1.0, last_.span);
        const double newSpan = std::max(1.0, o.span);
        const double ratio = newSpan / oldSpan;

        if (dtMs <= 0.0 || dtMs > maxGapMs_ ||
            ratio < minScaleRatio_ || ratio > maxScaleRatio_) {
            start(o);
            return false;
        }

        const double dt = dtMs / 1000.0;
        const double predX = last_.x + vx_ * dt;
        const double predY = last_.y + vy_ * dt;
        const double speed = std::hypot(vx_, vy_);
        const double radius = std::max(baseRadius_, spanFactor_ * std::max(oldSpan, newSpan)) +
                              speed * dt * speedAllowance_;
        const double dist = std::hypot(o.x - predX, o.y - predY);

        if (dist > radius) {
            start(o);
            return false;
        }

        const double instVx = (o.x - last_.x) / dt;
        const double instVy = (o.y - last_.y) / dt;
        if (hits_ <= 1) {
            vx_ = instVx;
            vy_ = instVy;
        } else {
            vx_ = velocityKeep_ * vx_ + (1.0 - velocityKeep_) * instVx;
            vy_ = velocityKeep_ * vy_ + (1.0 - velocityKeep_) * instVy;
        }

        last_ = o;
        ++hits_;
        return true;
    }

    bool ready(int requiredHits) const {
        return have_ && hits_ >= std::max(1, requiredHits);
    }

    int hits() const { return hits_; }
    bool active() const { return have_; }
    double velocityX() const { return vx_; }
    double velocityY() const { return vy_; }
    const HObservation& last() const { return last_; }
    HAcquisitionMethod lastMethod() const { return last_.method; }

    double ageMs(double nowMs) const {
        return have_ ? std::max(0.0, nowMs - last_.timeMs) : 1.0e12;
    }

private:
    void start(const HObservation& o) {
        have_ = true;
        hits_ = 1;
        vx_ = 0.0;
        vy_ = 0.0;
        last_ = o;
    }

    bool have_ = false;
    int hits_ = 0;
    double vx_ = 0.0;
    double vy_ = 0.0;
    HObservation last_{};

    // Deliberately generous enough for low processed FPS / fast camera motion,
    // while scale continuity and detector-backed observations still gate hits.
    double maxGapMs_ = 900.0;
    double minScaleRatio_ = 0.45;
    double maxScaleRatio_ = 2.20;
    double baseRadius_ = 14.0;
    double spanFactor_ = 2.00;
    double speedAllowance_ = 1.35;
    double velocityKeep_ = 0.55;
};

} // namespace helipad
