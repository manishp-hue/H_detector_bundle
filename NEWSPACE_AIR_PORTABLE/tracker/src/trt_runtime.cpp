#include "trt_runtime.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#ifdef HELIPAD_USE_TENSORRT
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#endif

namespace helipad {

#ifdef HELIPAD_USE_TENSORRT
namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << '\n';
    }
};

size_t dtypeBytes(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kBOOL: return 1;
        default: return 4;
    }
}

size_t volume(const nvinfer1::Dims& d) {
    size_t v=1;
    for (int i=0;i<d.nbDims;++i) {
        if (d.d[i] <= 0) throw std::runtime_error("Dynamic/invalid TensorRT shape. Build fixed-shape engines for this runtime.");
        v*=static_cast<size_t>(d.d[i]);
    }
    return v;
}

std::vector<char> readFile(const std::string& p) {
    std::ifstream f(p,std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open TensorRT plan: "+p);
    f.seekg(0,std::ios::end); auto n=f.tellg(); f.seekg(0,std::ios::beg);
    std::vector<char> b(static_cast<size_t>(n)); f.read(b.data(),n); return b;
}

struct TensorBuf {
    std::string name;
    nvinfer1::DataType dtype{};
    nvinfer1::Dims dims{};
    bool input=false;
    void* device=nullptr;
    size_t count=0;
    ~TensorBuf(){ if(device) cudaFree(device); }
};

class PlanRunner {
public:
    ~PlanRunner() {
        if(stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    bool load(const std::string& plan) {
        try {
            if(!stream_) {
                const cudaError_t e =
                    cudaStreamCreateWithFlags(
                        &stream_,
                        cudaStreamNonBlocking);

                if(e != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("cudaStreamCreate failed: ") +
                        cudaGetErrorString(e));
                }
            }

            auto bytes=readFile(plan);
            runtime_.reset(nvinfer1::createInferRuntime(logger_));
            if(!runtime_) return false;
            engine_.reset(runtime_->deserializeCudaEngine(bytes.data(),bytes.size()));
            if(!engine_) return false;
            ctx_.reset(engine_->createExecutionContext());
            if(!ctx_) return false;
            tensors_.clear();
#if NV_TENSORRT_MAJOR >= 10
            int n=engine_->getNbIOTensors();
            for(int i=0;i<n;++i){
                const char* nm=engine_->getIOTensorName(i);
                auto tb=std::make_unique<TensorBuf>(); tb->name=nm; tb->dtype=engine_->getTensorDataType(nm); tb->dims=engine_->getTensorShape(nm); tb->input=engine_->getTensorIOMode(nm)==nvinfer1::TensorIOMode::kINPUT; tb->count=volume(tb->dims);
                if(cudaMalloc(&tb->device,tb->count*dtypeBytes(tb->dtype))!=cudaSuccess) throw std::runtime_error("cudaMalloc failed");
                ctx_->setTensorAddress(nm,tb->device); tensors_[tb->name]=std::move(tb);
            }
#else
            int n=engine_->getNbBindings(); bindings_.resize(n,nullptr);
            for(int i=0;i<n;++i){
                auto tb=std::make_unique<TensorBuf>(); tb->name=engine_->getBindingName(i); tb->dtype=engine_->getBindingDataType(i); tb->dims=engine_->getBindingDimensions(i); tb->input=engine_->bindingIsInput(i); tb->count=volume(tb->dims);
                if(cudaMalloc(&tb->device,tb->count*dtypeBytes(tb->dtype))!=cudaSuccess) throw std::runtime_error("cudaMalloc failed"); bindings_[i]=tb->device; tensors_[tb->name]=std::move(tb);
            }
#endif
            return true;
        } catch(const std::exception& e){ std::cerr << "[TensorRT] load failed: " << e.what() << '\n'; return false; }
    }

    bool has(const std::string& n) const { return tensors_.count(n)!=0; }
    std::vector<std::string> names(bool inputs) const { std::vector<std::string> r; for(auto&kv:tensors_) if(kv.second->input==inputs) r.push_back(kv.first); return r; }
    std::vector<int> shape(const std::string& n) const { std::vector<int> s; auto it=tensors_.find(n); if(it==tensors_.end())return s; for(int i=0;i<it->second->dims.nbDims;++i)s.push_back(it->second->dims.d[i]); return s; }

    bool setFloat(const std::string& n,const std::vector<float>& v){
        auto it=tensors_.find(n); if(it==tensors_.end()||!it->second->input||v.size()!=it->second->count)return false; auto&t=*it->second;
        if(t.dtype==nvinfer1::DataType::kFLOAT) return cudaMemcpy(t.device,v.data(),v.size()*4,cudaMemcpyHostToDevice)==cudaSuccess;
        if(t.dtype==nvinfer1::DataType::kHALF){std::vector<__half> h(v.size());for(size_t i=0;i<v.size();++i)h[i]=__float2half(v[i]);return cudaMemcpy(t.device,h.data(),h.size()*2,cudaMemcpyHostToDevice)==cudaSuccess;}
        return false;
    }

    std::vector<float> getFloat(const std::string& n){
        auto it=tensors_.find(n);if(it==tensors_.end()||it->second->input)return{};auto&t=*it->second;std::vector<float> v(t.count);
        if(t.dtype==nvinfer1::DataType::kFLOAT){if(cudaMemcpy(v.data(),t.device,v.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess)return{};return v;}
        if(t.dtype==nvinfer1::DataType::kHALF){std::vector<__half> h(t.count);if(cudaMemcpy(h.data(),t.device,h.size()*2,cudaMemcpyDeviceToHost)!=cudaSuccess)return{};for(size_t i=0;i<v.size();++i)v[i]=__half2float(h[i]);return v;}
        return{};
    }

    bool run(){
        if(!stream_)
            return false;

#if NV_TENSORRT_MAJOR >= 10
        if(!ctx_->enqueueV3(stream_))
            return false;
#else
        if(!ctx_->enqueueV2(
                bindings_.data(),
                stream_,
                nullptr))
            return false;
#endif

        return cudaStreamSynchronize(stream_) ==
               cudaSuccess;
    }

private:
    struct RuntimeDel{void operator()(nvinfer1::IRuntime*p)const{if(!p)return;
#if NV_TENSORRT_MAJOR >= 10
        delete p;
#else
        p->destroy();
#endif
    }};
    struct EngineDel{void operator()(nvinfer1::ICudaEngine*p)const{if(!p)return;
#if NV_TENSORRT_MAJOR >= 10
        delete p;
#else
        p->destroy();
#endif
    }};
    struct ContextDel{void operator()(nvinfer1::IExecutionContext*p)const{if(!p)return;
#if NV_TENSORRT_MAJOR >= 10
        delete p;
#else
        p->destroy();
#endif
    }};
    Logger logger_;

    // Dedicated TensorRT stream.
    // Never run inference on CUDA stream 0.
    cudaStream_t stream_ = nullptr;

    std::unique_ptr<nvinfer1::IRuntime,RuntimeDel> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine,EngineDel> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext,ContextDel> ctx_;
    std::unordered_map<std::string,std::unique_ptr<TensorBuf>> tensors_;
#if NV_TENSORRT_MAJOR < 10
    std::vector<void*> bindings_;
#endif
};

std::vector<float> preprocessRgbNchw(const cv::Mat& bgr,int outW,int outH,float& scale,int& padX,int& padY){
    scale=std::min(float(outW)/bgr.cols,float(outH)/bgr.rows);int nw=std::max(1,int(std::round(bgr.cols*scale))),nh=std::max(1,int(std::round(bgr.rows*scale)));padX=(outW-nw)/2;padY=(outH-nh)/2;
    cv::Mat rs;cv::resize(bgr,rs,{nw,nh});cv::Mat canvas(outH,outW,CV_8UC3,cv::Scalar(114,114,114));rs.copyTo(canvas(cv::Rect(padX,padY,nw,nh)));cv::cvtColor(canvas,canvas,cv::COLOR_BGR2RGB);
    std::vector<float> x(3*outW*outH);const float mean[3]={.485f,.456f,.406f},stdv[3]={.229f,.224f,.225f};
    for(int y=0;y<outH;++y){const cv::Vec3b*row=canvas.ptr<cv::Vec3b>(y);for(int xx=0;xx<outW;++xx)for(int c=0;c<3;++c)x[c*outW*outH+y*outW+xx]=(row[xx][c]/255.0f-mean[c])/stdv[c];}
    return x;
}

} // namespace
#endif

struct TensorRtAiRecovery::Impl {
#ifdef HELIPAD_USE_TENSORRT
    PlanRunner dino, sam;
#endif
    bool dinoOk=false,samOk=false;
};

TensorRtAiRecovery::TensorRtAiRecovery():impl_(std::make_unique<Impl>()){}
TensorRtAiRecovery::~TensorRtAiRecovery()=default;

bool TensorRtAiRecovery::load(const std::string& dinoPlan,const std::string& samPlan){
#ifdef HELIPAD_USE_TENSORRT
    impl_->dinoOk=!dinoPlan.empty()&&impl_->dino.load(dinoPlan);
    impl_->samOk=!samPlan.empty()&&impl_->sam.load(samPlan);
    return impl_->dinoOk;
#else
    (void)dinoPlan;(void)samPlan;std::cerr<<"[AI] Built without TensorRT; geometry-only runtime active.\n";return false;
#endif
}

bool TensorRtAiRecovery::available() const{return impl_->dinoOk;}
bool TensorRtAiRecovery::samAvailable() const{return impl_->samOk;}

std::vector<AiProposal> TensorRtAiRecovery::detect(const cv::Mat& bgr,int topK){
    std::vector<AiProposal> out;
#ifdef HELIPAD_USE_TENSORRT
    if(!impl_->dinoOk||bgr.empty())return out;
    auto shp=impl_->dino.shape("images");if(shp.size()!=4)return out;int H=shp[2],W=shp[3];float scale;int px,py;auto input=preprocessRgbNchw(bgr,W,H,scale,px,py);
    if(!impl_->dino.setFloat("images",input)||!impl_->dino.run())return out;auto boxes=impl_->dino.getFloat("boxes"),scores=impl_->dino.getFloat("scores");int n=std::min<int>(scores.size(),boxes.size()/4);n=std::min(n,topK);
    for(int i=0;i<n;++i){float s=scores[i];if(s<.14f)continue;float x1=boxes[4*i],y1=boxes[4*i+1],x2=boxes[4*i+2],y2=boxes[4*i+3];
        // Export contract is normalized xyxy in the letterboxed image.
        x1=(x1*W-px)/scale;y1=(y1*H-py)/scale;x2=(x2*W-px)/scale;y2=(y2*H-py)/scale;cv::Rect2f r(x1,y1,x2-x1,y2-y1);r=clampRect(r,bgr.cols,bgr.rows);if(r.area()>4)out.push_back({r,s});}
#endif
    return out;
}

cv::Rect2f TensorRtAiRecovery::refineWithSam(const cv::Mat& bgr,const cv::Rect2f& proposal,float* maskScore){
    if(maskScore)*maskScore=0.0f;
#ifdef HELIPAD_USE_TENSORRT
    if(!impl_->samOk||bgr.empty())return proposal;
    // Optional SAM TensorRT contract:
    // input "image" [1,3,H,W], input "box" [1,4] normalized xyxy,
    // output "mask" [1,1,H,W], optional output "score" [1].
    auto shp=impl_->sam.shape("image");if(shp.size()!=4||!impl_->sam.has("box")||!impl_->sam.has("mask"))return proposal;int H=shp[2],W=shp[3];float scale;int px,py;auto input=preprocessRgbNchw(bgr,W,H,scale,px,py);
    cv::Rect2f p=clampRect(proposal,bgr.cols,bgr.rows);std::vector<float> box={ (p.x*scale+px)/W,(p.y*scale+py)/H,((p.x+p.width)*scale+px)/W,((p.y+p.height)*scale+py)/H };
    if(!impl_->sam.setFloat("image",input)||!impl_->sam.setFloat("box",box)||!impl_->sam.run())return proposal;auto mask=impl_->sam.getFloat("mask");if(mask.size()!=size_t(H*W))return proposal;auto sc=impl_->sam.getFloat("score");if(maskScore&& !sc.empty())*maskScore=sc[0];
    int minx=W,miny=H,maxx=-1,maxy=-1;for(int y=0;y<H;++y)for(int x=0;x<W;++x)if(mask[y*W+x]>.0f){minx=std::min(minx,x);miny=std::min(miny,y);maxx=std::max(maxx,x);maxy=std::max(maxy,y);}if(maxx<minx||maxy<miny)return proposal;
    float x1=(minx-px)/scale,y1=(miny-py)/scale,x2=(maxx-px)/scale,y2=(maxy-py)/scale;return clampRect({x1,y1,x2-x1,y2-y1},bgr.cols,bgr.rows);
#else
    (void)bgr;
#endif
    return proposal;
}

} // namespace helipad
