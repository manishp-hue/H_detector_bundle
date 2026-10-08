#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace helipad {

// ============================================================
// V15.26 SAFE APPROACH MATH
//
// Landing / recovery / docking / simulation state estimator.
//
// This module produces CAMERA-FRAME measurements only.
// It intentionally produces NO actuator commands.
// ============================================================

struct SafeApproachConfig
{
    // Replace with calibrated camera value.
    double hfovDeg = 70.0;

    // <= 0 means derive from aspect ratio.
    double vfovDeg = 0.0;

    // IMPORTANT:
    // These dimensions must correspond to the physical object
    // represented by Candidate::ringBox.
    //
    // Current simulation assumption:
    // helipad / ring footprint approximately 5 x 4 m.
    double knownRingWidthM  = 5.0;
    double knownRingHeightM = 4.0;

    // Alpha-beta state filter.
    double alphaPosition = 0.58;
    double betaPosition  = 0.10;

    double alphaScale = 0.50;
    double betaScale  = 0.08;

    // Ring must have useful pixel size before metric range
    // is considered meaningful.
    double minRingPixels = 12.0;

    // Reject strongly inconsistent width/height range estimates.
    double minRangeConsistency = 0.45;

    // Protect derivative calculations.
    double minDt = 0.001;
    double maxDt = 0.20;
};


struct SafeApproachMeasurement
{
    cv::Point2f center{};
    cv::Rect2f trackBox{};

    std::optional<cv::Rect2f> ringBox;

    float detectorScore = 0.0f;
    float flowQuality   = 0.0f;

    bool ringVerified = false;
};


struct SafeApproachResult
{
    bool valid = false;

    // ---------------- image error ----------------

    double errorX = 0.0;
    double errorY = 0.0;

    // ---------------- camera LOS ----------------

    double bearingXDeg = 0.0;
    double bearingYDeg = 0.0;

    double losRateXDegS = 0.0;
    double losRateYDegS = 0.0;

    // ---------------- apparent size ----------------

    double scalePx = 0.0;
    double scaleRatePxS = 0.0;

    // ---------------- time-to-contact ----------------

    bool tauValid = false;
    double tauS =
        std::numeric_limits<double>::quiet_NaN();

    // ---------------- metric size-range ----------------

    bool rangeValid = false;

    double rangeWidthM =
        std::numeric_limits<double>::quiet_NaN();

    double rangeHeightM =
        std::numeric_limits<double>::quiet_NaN();

    double sizeRangeM =
        std::numeric_limits<double>::quiet_NaN();

    double rangeConsistency = 0.0;

    // ---------------- quality ----------------

    double measurementQuality = 0.0;
};


class SafeApproachEstimator
{
public:

    explicit SafeApproachEstimator(
        const SafeApproachConfig& cfg = {})
        : cfg_(cfg)
    {
    }


    void reset()
    {
        initialized_ = false;

        ex_ = 0.0;
        ey_ = 0.0;

        vex_ = 0.0;
        vey_ = 0.0;

        scale_ = 0.0;
        scaleRate_ = 0.0;

        missing_ = 0;
    }


    SafeApproachResult update(
        int frameWidth,
        int frameHeight,
        const std::optional<SafeApproachMeasurement>& measurement,
        double dt)
    {
        SafeApproachResult out;


        if(frameWidth <= 0 ||
           frameHeight <= 0)
        {
            return out;
        }


        if(!measurement)
        {
            ++missing_;

            // Do not expose predicted guidance after target
            // ownership has been lost.
            if(missing_ >= 3)
                reset();

            return out;
        }


        missing_ = 0;


        dt =
            std::clamp(
                dt,
                cfg_.minDt,
                cfg_.maxDt);


        const auto& m =
            *measurement;


        // ====================================================
        // 1. NORMALIZED IMAGE ERROR
        //
        // -1 ... +1 approximately spans image width/height.
        //
        // x positive = target right
        // y positive = target below camera centre
        // ====================================================

        const double mx =
            (double(m.center.x) -
             0.5 * frameWidth) /
            (0.5 * frameWidth);


        const double my =
            (double(m.center.y) -
             0.5 * frameHeight) /
            (0.5 * frameHeight);


        // ====================================================
        // 2. APPARENT TARGET SCALE
        //
        // sqrt(area) is less sensitive than width alone to
        // modest aspect-ratio changes.
        // ====================================================

        cv::Rect2f scaleBox =
            m.trackBox;


        if(m.ringVerified &&
           m.ringBox &&
           m.ringBox->width > 1.0f &&
           m.ringBox->height > 1.0f)
        {
            scaleBox =
                *m.ringBox;
        }


        const double measuredScale =
            std::sqrt(
                std::max(
                    1.0,
                    double(scaleBox.width) *
                    double(scaleBox.height)));


        // ====================================================
        // 3. MEASUREMENT QUALITY
        //
        // This is a state-estimation weight,
        // NOT an actuator authority command.
        // ====================================================

        const double detectorQ =
            std::clamp(
                double(m.detectorScore),
                0.0,
                1.0);


        // Optical-flow quality becomes "strong" around 0.4.
        const double flowQ =
            std::clamp(
                double(m.flowQuality) / 0.40,
                0.0,
                1.0);


        const double geometryQ =
            m.ringVerified
                ? 1.0
                : 0.65;


        const double quality =
            std::clamp(
                0.55 * detectorQ +
                0.30 * flowQ +
                0.15 * geometryQ,
                0.0,
                1.0);


        // ====================================================
        // 4. ALPHA-BETA FILTER
        //
        // State:
        //
        // [ ex ey vex vey scale scaleRate ]
        //
        // Detector gives measurement.
        // Filter gives prediction between detector corrections.
        // ====================================================

        if(!initialized_)
        {
            ex_ = mx;
            ey_ = my;

            vex_ = 0.0;
            vey_ = 0.0;

            scale_ = measuredScale;
            scaleRate_ = 0.0;

            initialized_ = true;
        }
        else
        {
            // ---------- predict ----------

            const double predX =
                ex_ + vex_ * dt;

            const double predY =
                ey_ + vey_ * dt;

            const double predScale =
                std::max(
                    1.0,
                    scale_ +
                    scaleRate_ * dt);


            // ---------- residual ----------

            const double rx =
                mx - predX;

            const double ry =
                my - predY;

            const double rs =
                measuredScale -
                predScale;


            // Strong measurement -> slightly stronger update.
            const double qScale =
                0.75 +
                0.40 * quality;


            const double alphaPos =
                std::clamp(
                    cfg_.alphaPosition * qScale,
                    0.25,
                    0.85);


            const double betaPos =
                std::clamp(
                    cfg_.betaPosition * qScale,
                    0.02,
                    0.30);


            const double alphaScale =
                std::clamp(
                    cfg_.alphaScale * qScale,
                    0.20,
                    0.80);


            const double betaScale =
                std::clamp(
                    cfg_.betaScale * qScale,
                    0.01,
                    0.25);


            // ---------- update ----------

            ex_ =
                predX +
                alphaPos * rx;

            ey_ =
                predY +
                alphaPos * ry;


            vex_ +=
                betaPos *
                rx / dt;

            vey_ +=
                betaPos *
                ry / dt;


            scale_ =
                std::max(
                    1.0,
                    predScale +
                    alphaScale * rs);


            scaleRate_ +=
                betaScale *
                rs / dt;
        }


        // ====================================================
        // 5. CAMERA INTRINSICS FROM FOV
        //
        // For production, calibrated fx/fy are preferable.
        // ====================================================

        constexpr double PI =
            3.14159265358979323846;


        const auto deg2rad =
            [&](double d)
            {
                return d * PI / 180.0;
            };


        const auto rad2deg =
            [&](double r)
            {
                return r * 180.0 / PI;
            };


        const double halfHFov =
            0.5 *
            deg2rad(
                cfg_.hfovDeg);


        const double tanHalfH =
            std::tan(
                halfHFov);


        double tanHalfV = 0.0;


        if(cfg_.vfovDeg > 0.0)
        {
            tanHalfV =
                std::tan(
                    0.5 *
                    deg2rad(
                        cfg_.vfovDeg));
        }
        else
        {
            // Square-pixel approximation.
            tanHalfV =
                (double(frameHeight) /
                 double(frameWidth)) *
                tanHalfH;
        }


        const double fx =
            double(frameWidth) /
            (2.0 * tanHalfH);


        const double fy =
            double(frameHeight) /
            (2.0 * tanHalfV);


        // ====================================================
        // 6. NORMALIZED ERROR -> CAMERA BEARING
        // ====================================================

        const double tx =
            ex_ *
            tanHalfH;


        const double ty =
            ey_ *
            tanHalfV;


        const double bearingX =
            std::atan(tx);


        const double bearingY =
            std::atan(ty);


        // ====================================================
        // 7. LOS ANGULAR RATE
        //
        // If:
        //
        // theta = atan(e * tan(FOV/2))
        //
        // then:
        //
        // dtheta/dt =
        //   tan(FOV/2)/(1 + (...^2)) * de/dt
        //
        // This is cleaner than differencing noisy angles.
        // ====================================================

        const double losRateX =
            (tanHalfH /
             (1.0 + tx*tx)) *
            vex_;


        const double losRateY =
            (tanHalfV /
             (1.0 + ty*ty)) *
            vey_;


        // ====================================================
        // 8. TIME-TO-CONTACT FROM IMAGE EXPANSION
        //
        // tau = scale / scaleRate
        //
        // Only valid while target is actually expanding.
        // ====================================================

        if(scaleRate_ > 0.50)
        {
            const double tau =
                scale_ /
                scaleRate_;


            if(std::isfinite(tau) &&
               tau > 0.0)
            {
                out.tauValid = true;
                out.tauS = tau;
            }
        }


        // ====================================================
        // 9. KNOWN-SIZE RING RANGE
        //
        // R ~= f * D / d
        //
        // Only actual verified RingH geometry may create
        // a metric size-based range.
        // ====================================================

        if(m.ringVerified &&
           m.ringBox &&
           m.ringBox->width >= cfg_.minRingPixels &&
           m.ringBox->height >= cfg_.minRingPixels &&
           cfg_.knownRingWidthM > 0.0 &&
           cfg_.knownRingHeightM > 0.0)
        {
            const double rw =
                fx *
                cfg_.knownRingWidthM /
                double(m.ringBox->width);


            const double rh =
                fy *
                cfg_.knownRingHeightM /
                double(m.ringBox->height);


            if(rw > 0.0 &&
               rh > 0.0 &&
               std::isfinite(rw) &&
               std::isfinite(rh))
            {
                out.rangeWidthM = rw;
                out.rangeHeightM = rh;


                out.rangeConsistency =
                    std::min(rw,rh) /
                    std::max(rw,rh);


                // Geometric mean reduces domination by one
                // badly foreshortened axis.
                out.sizeRangeM =
                    std::sqrt(
                        rw * rh);


                out.rangeValid =
                    out.rangeConsistency >=
                    cfg_.minRangeConsistency;
            }
        }


        out.valid = true;

        out.errorX = ex_;
        out.errorY = ey_;

        out.bearingXDeg =
            rad2deg(
                bearingX);

        out.bearingYDeg =
            rad2deg(
                bearingY);

        out.losRateXDegS =
            rad2deg(
                losRateX);

        out.losRateYDegS =
            rad2deg(
                losRateY);

        out.scalePx = scale_;
        out.scaleRatePxS = scaleRate_;

        out.measurementQuality =
            quality;


        return out;
    }


    // ========================================================
    // Optional safe-approach utility.
    //
    // Requires HORIZONTAL range and altitude from a separate
    // trusted navigation source.
    //
    // Positive:
    // aircraft above requested glide path.
    //
    // Negative:
    // aircraft below requested glide path.
    // ========================================================

    static std::optional<double> glidePathErrorM(
        double altitudeM,
        double horizontalRangeM,
        double desiredApproachDeg)
    {
        if(altitudeM < 0.0 ||
           horizontalRangeM <= 0.0)
        {
            return std::nullopt;
        }


        constexpr double PI =
            3.14159265358979323846;


        const double desiredAltitude =
            horizontalRangeM *
            std::tan(
                desiredApproachDeg *
                PI / 180.0);


        return
            altitudeM -
            desiredAltitude;
    }


private:

    SafeApproachConfig cfg_;

    bool initialized_ = false;

    int missing_ = 0;

    double ex_ = 0.0;
    double ey_ = 0.0;

    double vex_ = 0.0;
    double vey_ = 0.0;

    double scale_ = 0.0;
    double scaleRate_ = 0.0;
};


} // namespace helipad
