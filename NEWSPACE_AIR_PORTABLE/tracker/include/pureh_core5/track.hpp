#pragma once

// V15.56 header-only integration of PureH_CoreLock_05 StrictTracker.
// Function bodies are copied from the repository's track.cpp; only `inline`
// linkage was added so the original tracker can be used without changing
// Aryan's CMake source list. Detection/tracking behavior is unchanged.

#include <opencv2/opencv.hpp>

class StrictTracker {
public:
    bool initialize(const cv::Mat& frame_bgr, cv::Rect detectedTarget);
    bool update(const cv::Mat& frame_bgr, cv::Rect& drawTarget, cv::Rect& drawAnchor);
    bool isActive() const;
    void reset();
    double targetConfidence() const;
    double anchorConfidence() const;

private:
    cv::Rect clampRect(cv::Rect r, cv::Size s);
    cv::Rect scaleRect(cv::Rect r, float scale);
    cv::Rect unscaleRect(cv::Rect r, float scale);
    cv::Rect expandRect(cv::Rect r, float factor, cv::Size s);

    void stabilizeFrame(const cv::Mat& src, cv::Mat& dst);
    cv::Rect autoSelectAnchor(const cv::Mat& frame, cv::Rect targetBox);

    double templateConfidence(const cv::Mat& frame, cv::Rect box, const cv::Mat& templ);
    bool boxTouchesBorder(cv::Rect box, cv::Size size, int margin);
    bool reanchorAroundTarget(const cv::Mat& frame_bgr, const cv::Mat& processed);

    bool localSearchTemplate(
        const cv::Mat& frame,
        const cv::Mat& templ,
        cv::Rect predictedBox,
        cv::Rect& correctedBox,
        int searchMargin,
        double minScore
    );

    cv::Rect recoverTargetFromAnchor(
        cv::Rect anchorBox,
        cv::Point2f relCenter,
        cv::Size2f relSize
    );

private:
    static constexpr float TRACK_SCALE = 0.5f;

    static constexpr double TARGET_MIN_CONF = 0.30;
    static constexpr double ANCHOR_MIN_CONF = 0.34;

    static constexpr int TRACK_FAIL_LIMIT = 18;
    static constexpr int BORDER_FAIL_LIMIT = 3;

    bool active = false;
    int failCounter = 0;
    int borderFailCounter = 0;

    cv::Rect targetBox;
    cv::Rect anchorBox;

    cv::Mat targetTemplate;
    cv::Mat anchorTemplate;

    cv::Point2f relCenter;
    cv::Size2f relSize;

    double lastTargetConfidence_ = 0.0;
    double lastAnchorConfidence_ = 0.0;
};

// ---- Upstream PureH_CoreLock_05 track.cpp (inline linkage only) ----
#include <algorithm>
#include <cmath>

inline cv::Rect StrictTracker::clampRect(cv::Rect r, cv::Size s) {
    return r & cv::Rect(0, 0, s.width, s.height);
}

inline cv::Rect StrictTracker::scaleRect(cv::Rect r, float scale) {
    return cv::Rect(
        int(r.x * scale),
        int(r.y * scale),
        int(r.width * scale),
        int(r.height * scale)
    );
}

inline cv::Rect StrictTracker::unscaleRect(cv::Rect r, float scale) {
    return cv::Rect(
        int(r.x / scale),
        int(r.y / scale),
        int(r.width / scale),
        int(r.height / scale)
    );
}

inline cv::Rect StrictTracker::expandRect(cv::Rect r, float factor, cv::Size s) {
    cv::Point c(r.x + r.width / 2, r.y + r.height / 2);

    int nw = std::max(8, int(r.width * factor));
    int nh = std::max(8, int(r.height * factor));

    cv::Rect out(c.x - nw / 2, c.y - nh / 2, nw, nh);
    return clampRect(out, s);
}

inline void StrictTracker::stabilizeFrame(const cv::Mat& src, cv::Mat& dst) {
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(8, 8));

    // The guidance pipeline is grayscale-only. Keep the original tracker logic,
    // but avoid converting gray -> BGR -> Lab -> gray on every frame.
    if (src.channels() == 1) {
        clahe->apply(src, dst);
        cv::GaussianBlur(dst, dst, cv::Size(3, 3), 0);
        return;
    }

    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> ch;
    cv::split(lab, ch);
    clahe->apply(ch[0], ch[0]);

    cv::merge(ch, lab);
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
    cv::GaussianBlur(dst, dst, cv::Size(3, 3), 0);
}

inline cv::Rect StrictTracker::autoSelectAnchor(const cv::Mat& frame, cv::Rect targetBox) {
    cv::Rect search = expandRect(targetBox, 2.2f, frame.size());

    cv::Mat roi = frame(search);
    cv::Mat gray, blurImg, edges;

    if (roi.channels() == 1) gray = roi;
    else cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurImg, cv::Size(5, 5), 0);

    cv::Canny(blurImg, edges, 40, 120);
    cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Point targetCenter(
        targetBox.x + targetBox.width / 2 - search.x,
        targetBox.y + targetBox.height / 2 - search.y
    );

    cv::Rect best;
    double bestScore = -1.0;

    for (auto& c : contours) {
        cv::Rect b = cv::boundingRect(c);

        if (!b.contains(targetCenter))
            continue;

        if (b.width < targetBox.width * 1.05 ||
            b.height < targetBox.height * 1.05)
            continue;

        if (b.width > targetBox.width * 2.8 ||
            b.height > targetBox.height * 2.8)
            continue;

        double aspect = double(b.width) / std::max(1, b.height);
        if (aspect < 0.45 || aspect > 1.8)
            continue;

        double areaRatio = double(b.area()) / std::max(1, targetBox.area());
        double score = 1000.0 - std::abs(areaRatio - 1.8) * 200.0;

        if (score > bestScore) {
            bestScore = score;
            best = b;
        }
    }

    if (best.area() > 0) {
        best.x += search.x;
        best.y += search.y;

        best = expandRect(best, 1.08f, frame.size());
        return clampRect(best, frame.size());
    }

    return expandRect(targetBox, 1.35f, frame.size());
}

inline double StrictTracker::templateConfidence(
    const cv::Mat& frame,
    cv::Rect box,
    const cv::Mat& templ
) {
    box = clampRect(box, frame.size());

    if (box.width < 8 || box.height < 8 || templ.empty())
        return 0.0;

    cv::Mat gray, roi, temp;
    if (frame.channels() == 1) gray = frame;
    else cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    roi = gray(box);
    cv::resize(templ, temp, roi.size());

    cv::Mat result;
    cv::matchTemplate(roi, temp, result, cv::TM_CCOEFF_NORMED);

    double maxVal = 0.0;
    cv::minMaxLoc(result, nullptr, &maxVal);

    return maxVal;
}

inline bool StrictTracker::boxTouchesBorder(cv::Rect box, cv::Size size, int margin) {
    if (box.area() <= 0) return true;

    return box.x <= margin ||
           box.y <= margin ||
           box.x + box.width >= size.width - margin ||
           box.y + box.height >= size.height - margin;
}

inline bool StrictTracker::reanchorAroundTarget(
    const cv::Mat& frame_bgr,
    const cv::Mat& processed
) {
    cv::Rect targetFull = clampRect(unscaleRect(targetBox, TRACK_SCALE), frame_bgr.size());
    if (targetFull.area() <= 0) return false;

    cv::Rect anchorFull = autoSelectAnchor(frame_bgr, targetFull);
    if (anchorFull.area() <= 0) return false;

    cv::Rect newAnchor = clampRect(scaleRect(anchorFull, TRACK_SCALE), processed.size());
    if (newAnchor.area() <= 0) return false;

    relCenter = cv::Point2f(
        float(targetBox.x + targetBox.width * 0.5f - newAnchor.x) / newAnchor.width,
        float(targetBox.y + targetBox.height * 0.5f - newAnchor.y) / newAnchor.height
    );

    relSize = cv::Size2f(
        float(targetBox.width) / newAnchor.width,
        float(targetBox.height) / newAnchor.height
    );

    cv::Mat gray;
    if (processed.channels() == 1) gray = processed;
    else cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

    anchorBox = newAnchor;
    anchorTemplate = gray(anchorBox).clone();

    return true;
}

inline bool StrictTracker::localSearchTemplate(
    const cv::Mat& frame,
    const cv::Mat& templ,
    cv::Rect predictedBox,
    cv::Rect& correctedBox,
    int searchMargin,
    double minScore
) {
    if (templ.empty()) return false;

    cv::Mat gray;
    if (frame.channels() == 1) gray = frame;
    else cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Rect searchArea(
        predictedBox.x - searchMargin,
        predictedBox.y - searchMargin,
        predictedBox.width + searchMargin * 2,
        predictedBox.height + searchMargin * 2
    );

    searchArea = clampRect(searchArea, frame.size());

    if (searchArea.width <= templ.cols ||
        searchArea.height <= templ.rows)
        return false;

    cv::Mat result;
    cv::matchTemplate(gray(searchArea), templ, result, cv::TM_CCOEFF_NORMED);

    double maxVal = 0.0;
    cv::Point maxLoc;

    cv::minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

    if (maxVal < minScore)
        return false;

    correctedBox = cv::Rect(
        searchArea.x + maxLoc.x,
        searchArea.y + maxLoc.y,
        templ.cols,
        templ.rows
    );

    correctedBox = clampRect(correctedBox, frame.size());
    return correctedBox.area() > 0;
}

inline cv::Rect StrictTracker::recoverTargetFromAnchor(
    cv::Rect anchorBox,
    cv::Point2f relCenter,
    cv::Size2f relSize
) {
    int w = std::max(6, int(anchorBox.width * relSize.width));
    int h = std::max(6, int(anchorBox.height * relSize.height));

    int cx = int(anchorBox.x + relCenter.x * anchorBox.width);
    int cy = int(anchorBox.y + relCenter.y * anchorBox.height);

    return cv::Rect(cx - w / 2, cy - h / 2, w, h);
}

inline bool StrictTracker::initialize(const cv::Mat& frame_bgr, cv::Rect detectedTarget) {
    if (frame_bgr.empty()) return false;

    cv::Rect targetBoxFull = clampRect(detectedTarget, frame_bgr.size());

    if (targetBoxFull.area() <= 0)
        return false;

    cv::Rect anchorBoxFull = autoSelectAnchor(frame_bgr, targetBoxFull);

    if (anchorBoxFull.area() <= 0)
        return false;

    relCenter = cv::Point2f(
        float(targetBoxFull.x + targetBoxFull.width * 0.5f - anchorBoxFull.x) / anchorBoxFull.width,
        float(targetBoxFull.y + targetBoxFull.height * 0.5f - anchorBoxFull.y) / anchorBoxFull.height
    );

    relSize = cv::Size2f(
        float(targetBoxFull.width) / anchorBoxFull.width,
        float(targetBoxFull.height) / anchorBoxFull.height
    );

    cv::Mat small, processed;
    cv::resize(frame_bgr, small, cv::Size(), TRACK_SCALE, TRACK_SCALE);
    stabilizeFrame(small, processed);

    targetBox = clampRect(scaleRect(targetBoxFull, TRACK_SCALE), processed.size());
    anchorBox = clampRect(scaleRect(anchorBoxFull, TRACK_SCALE), processed.size());

    if (targetBox.area() <= 0 || anchorBox.area() <= 0)
        return false;

    cv::Mat gray;
    if (processed.channels() == 1) gray = processed;
    else cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

    targetTemplate = gray(targetBox).clone();
    anchorTemplate = gray(anchorBox).clone();

    failCounter = 0;
    borderFailCounter = 0;
    lastTargetConfidence_ = 1.0;
    lastAnchorConfidence_ = 1.0;
    active = true;

    return true;
}

inline bool StrictTracker::update(
    const cv::Mat& frame_bgr,
    cv::Rect& drawTarget,
    cv::Rect& drawAnchor
) {
    if (!active || frame_bgr.empty())
        return false;

    cv::Mat small, processed;
    cv::resize(frame_bgr, small, cv::Size(), TRACK_SCALE, TRACK_SCALE);
    stabilizeFrame(small, processed);

    cv::Rect newTarget;
    cv::Rect newAnchor;

    bool targetOK = localSearchTemplate(
        processed,
        targetTemplate,
        targetBox,
        newTarget,
        90,
        TARGET_MIN_CONF
    );

    bool anchorOK = localSearchTemplate(
        processed,
        anchorTemplate,
        anchorBox,
        newAnchor,
        120,
        ANCHOR_MIN_CONF
    );

    if (targetOK)
        targetBox = clampRect(newTarget, processed.size());

    if (anchorOK)
        anchorBox = clampRect(newAnchor, processed.size());

    double anchorConf = templateConfidence(processed, anchorBox, anchorTemplate);
    double targetConf = templateConfidence(processed, targetBox, targetTemplate);
    lastAnchorConfidence_ = anchorConf;
    lastTargetConfidence_ = targetConf;

    if (anchorConf < ANCHOR_MIN_CONF)
        anchorOK = false;

    if (targetConf < TARGET_MIN_CONF)
        targetOK = false;

    if (boxTouchesBorder(targetBox, processed.size(), 3) && targetConf < 0.55) {
        targetOK = false;
        borderFailCounter++;
    } else {
        borderFailCounter = 0;
    }

    if (!anchorOK && targetOK) {
        if (reanchorAroundTarget(frame_bgr, processed)) {
            anchorOK = true;
        }
    }

    if (!anchorOK) {
        cv::Rect correctedAnchor;

        bool foundAnchor = localSearchTemplate(
            processed,
            anchorTemplate,
            anchorBox,
            correctedAnchor,
            90,
            0.34
        );

        if (foundAnchor) {
            anchorBox = correctedAnchor;
            anchorOK = true;
        }
    }

    if (!targetOK && anchorOK) {
        cv::Rect recoveredTarget = recoverTargetFromAnchor(
            anchorBox,
            relCenter,
            relSize
        );

        recoveredTarget = clampRect(recoveredTarget, processed.size());

        if (!boxTouchesBorder(recoveredTarget, processed.size(), 3)) {
            cv::Rect correctedTarget;

            bool foundTarget = localSearchTemplate(
                processed,
                targetTemplate,
                recoveredTarget,
                correctedTarget,
                120,
                0.30
            );

            if (foundTarget && !boxTouchesBorder(correctedTarget, processed.size(), 3)) {
                targetBox = correctedTarget;
                targetOK = true;
            }
        }
    }

    bool keepAlive = targetOK || anchorOK;

    if (!keepAlive)
        failCounter++;
    else
        failCounter = 0;

    drawTarget = clampRect(unscaleRect(targetBox, TRACK_SCALE), frame_bgr.size());
    drawAnchor = clampRect(unscaleRect(anchorBox, TRACK_SCALE), frame_bgr.size());

    if (borderFailCounter >= BORDER_FAIL_LIMIT || failCounter >= TRACK_FAIL_LIMIT) {
        active = false;
        return false;
    }

    return targetOK;
}

inline bool StrictTracker::isActive() const {
    return active;
}

inline void StrictTracker::reset() {
    active = false;
    failCounter = 0;
    borderFailCounter = 0;
    targetBox = {};
    anchorBox = {};
    targetTemplate.release();
    anchorTemplate.release();
    relCenter = {};
    relSize = {};
    lastTargetConfidence_ = 0.0;
    lastAnchorConfidence_ = 0.0;
}

inline double StrictTracker::targetConfidence() const {
    return lastTargetConfidence_;
}

inline double StrictTracker::anchorConfidence() const {
    return lastAnchorConfidence_;
}
