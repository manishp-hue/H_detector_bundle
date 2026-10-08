#include <QApplication>
#include <QBoxLayout>
#include <QCloseEvent>
#include <QFrame>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QPushButton>
#include <QTcpSocket>
#include <QTimer>
#include <QStyle>
#include <QStatusBar>
#include <QResizeEvent>
#include <QDebug>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <stdexcept>

namespace {
using namespace std::chrono_literals;

std::int64_t monoMs(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Config{
    QString jetsonIp="192.168.144.16";
    int controlPort=5701;
    int rtspPort=8554;
};

bool parseArgs(int argc,char** argv,Config& c){
    for(int i=1;i<argc;++i){
        const QString k=argv[i];
        auto need=[&](const char* n)->QString{
            if(i+1>=argc)throw std::runtime_error(std::string("missing value for ")+n);
            return argv[++i];
        };
        if(k=="--jetson-ip")c.jetsonIp=need("--jetson-ip");
        else if(k=="--control-port")c.controlPort=need("--control-port").toInt();
        else if(k=="--rtsp-port")c.rtspPort=need("--rtsp-port").toInt();
        else if(k=="-h"||k=="--help"){
            qInfo("subscriber_node --jetson-ip 192.168.144.16 --control-port 5701 --rtsp-port 8554");
            return false;
        }else throw std::runtime_error(("unknown option: "+k).toStdString());
    }
    return true;
}

class VideoView:public QFrame{
    Q_OBJECT
    QLabel* label_=nullptr;
    QImage frame_;
public:
    explicit VideoView(QWidget* p=nullptr):QFrame(p){
        setObjectName("videoFrame");setMinimumSize(800,450);
        auto* l=new QVBoxLayout(this);l->setContentsMargins(0,0,0,0);
        label_=new QLabel("DETECTION OFF\n\nNo video transmitted",this);
        label_->setAlignment(Qt::AlignCenter);label_->setObjectName("videoLabel");l->addWidget(label_);
    }
    void setFrame(const QImage& img){frame_=img;refresh();}
    void clearToMessage(const QString& t){frame_=QImage();label_->setPixmap(QPixmap());label_->setText(t);}
protected:
    void resizeEvent(QResizeEvent* e)override{QFrame::resizeEvent(e);refresh();}
private:
    void refresh(){
        if(frame_.isNull())return;
        label_->setText(QString());
        label_->setPixmap(QPixmap::fromImage(frame_).scaled(label_->size(),Qt::KeepAspectRatio,Qt::FastTransformation));
    }
};

class GstRtspReceiver:public QObject{
    Q_OBJECT
    QTimer displayTimer_;
    std::mutex frameMutex_;
    QImage latest_;
    std::atomic<std::int64_t> lastDecodedMs_{0};
    std::atomic<std::int64_t> sessionStartMs_{0};

    std::mutex commandMutex_;
    std::condition_variable commandCv_;
    std::thread worker_;
    bool quit_=false;
    bool desired_=false;
    QString desiredUrl_;
    std::uint64_t generation_=0;
    std::atomic<bool> running_{false};
    std::atomic<bool> starting_{false};

    static GstFlowReturn sampleCb(GstAppSink* sink,gpointer data){
        return static_cast<GstRtspReceiver*>(data)->sample(sink);
    }

    GstFlowReturn sample(GstAppSink* sink){
        GstSample* s=gst_app_sink_pull_sample(sink);
        if(!s)return GST_FLOW_EOS;
        GstBuffer* b=gst_sample_get_buffer(s);
        GstCaps* caps=gst_sample_get_caps(s);
        GstVideoInfo vi;gst_video_info_init(&vi);GstVideoFrame vf{};
        if(!b||!caps||!gst_video_info_from_caps(&vi,caps)||
           GST_VIDEO_INFO_FORMAT(&vi)!=GST_VIDEO_FORMAT_RGB||
           !gst_video_frame_map(&vf,&vi,b,GST_MAP_READ)){
            gst_sample_unref(s);return GST_FLOW_OK;
        }
        const int stride=GST_VIDEO_FRAME_PLANE_STRIDE(&vf,0);
        if(stride>0){
            QImage view(static_cast<const uchar*>(GST_VIDEO_FRAME_PLANE_DATA(&vf,0)),
                        GST_VIDEO_INFO_WIDTH(&vi),GST_VIDEO_INFO_HEIGHT(&vi),stride,QImage::Format_RGB888);
            QImage owned=view.copy();
            std::lock_guard<std::mutex> l(frameMutex_);
            latest_=std::move(owned);
        }
        lastDecodedMs_.store(monoMs(),std::memory_order_release);
        gst_video_frame_unmap(&vf);gst_sample_unref(s);
        return GST_FLOW_OK;
    }

    bool stillWanted(std::uint64_t gen){
        std::lock_guard<std::mutex> l(commandMutex_);
        return !quit_ && desired_ && generation_==gen;
    }

    void publishError(const QString& text){
        qWarning()<<"[VIDEO]"<<text;
        emit streamError(text);
    }

    void workerLoop(){
        gst_init(nullptr,nullptr);
        for(;;){
            QString url;std::uint64_t gen=0;
            {
                std::unique_lock<std::mutex> l(commandMutex_);
                commandCv_.wait(l,[&]{return quit_||desired_;});
                if(quit_)break;
                url=desiredUrl_;gen=generation_;
            }

            starting_=true;running_=false;lastDecodedMs_=0;sessionStartMs_=monoMs();
            qInfo()<<"[VIDEO] RTSP worker connecting to"<<url;

            const QString pipe=QString(
                "rtspsrc name=rtsp location=\"%1\" protocols=tcp latency=0 buffer-mode=none drop-on-latency=true tcp-timeout=10000000 "
                "! rtph264depay ! h264parse ! video/x-h264,alignment=au "
                "! avdec_h264 max-threads=2 "
                "! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
                "! videoconvert n-threads=2 ! video/x-raw,format=RGB "
                "! appsink name=vsink emit-signals=true max-buffers=1 drop=true sync=false enable-last-sample=false"
            ).arg(url);

            GError* err=nullptr;
            GstElement* pipeline=gst_parse_launch(pipe.toUtf8().constData(),&err);
            if(err||!pipeline){
                const QString why=err?QString::fromUtf8(err->message):"Pipeline allocation failed";
                if(err)g_error_free(err);
                starting_=false;
                {std::lock_guard<std::mutex> l(commandMutex_);if(generation_==gen)desired_=false;}
                publishError(why);
                if(pipeline)gst_object_unref(pipeline);
                continue;
            }

            GstElement* rtsp=gst_bin_get_by_name(GST_BIN(pipeline),"rtsp");
            if(rtsp){
                if(g_object_class_find_property(G_OBJECT_GET_CLASS(rtsp),"tcp-timestamp"))
                    g_object_set(rtsp,"tcp-timestamp",FALSE,nullptr);
                gst_object_unref(rtsp);
            }
            GstAppSink* sink=GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline),"vsink"));
            GstBus* bus=gst_element_get_bus(pipeline);
            if(!sink||!bus){
                if(sink)gst_object_unref(sink);if(bus)gst_object_unref(bus);
                gst_element_set_state(pipeline,GST_STATE_NULL);gst_object_unref(pipeline);
                starting_=false;
                {std::lock_guard<std::mutex> l(commandMutex_);if(generation_==gen)desired_=false;}
                publishError("Missing GStreamer sink/bus");
                continue;
            }
            g_signal_connect(sink,"new-sample",G_CALLBACK(sampleCb),this);

            const auto change=gst_element_set_state(pipeline,GST_STATE_PLAYING);
            if(change==GST_STATE_CHANGE_FAILURE){
                gst_object_unref(sink);gst_object_unref(bus);
                gst_element_set_state(pipeline,GST_STATE_NULL);gst_object_unref(pipeline);
                starting_=false;
                {std::lock_guard<std::mutex> l(commandMutex_);if(generation_==gen)desired_=false;}
                publishError("RTSP startup failed");
                continue;
            }

            starting_=false;running_=true;
            qInfo()<<"[VIDEO] RTSP pipeline PLAYING";
            QString failure;
            while(stillWanted(gen)){
                GstMessage* m=gst_bus_timed_pop_filtered(
                    bus,100*GST_MSECOND,
                    GstMessageType(GST_MESSAGE_ERROR|GST_MESSAGE_EOS));
                if(m){
                    if(GST_MESSAGE_TYPE(m)==GST_MESSAGE_ERROR){
                        GError* e=nullptr;gchar* dbg=nullptr;
                        gst_message_parse_error(m,&e,&dbg);
                        failure=e?QString::fromUtf8(e->message):"Video error";
                        if(dbg)qWarning()<<"[VIDEO GST]"<<dbg;
                        if(e)g_error_free(e);g_free(dbg);
                    }else failure="Video EOS";
                    gst_message_unref(m);break;
                }
                const auto last=lastDecodedMs_.load(std::memory_order_acquire);
                const auto now=monoMs();
                if(last==0 && now-sessionStartMs_.load()>8000){failure="No first decoded frame within 8 seconds";break;}
                if(last>0 && now-last>1500){failure="No decoded frame for 1.5 seconds";break;}
            }

            running_=false;starting_=false;
            gst_element_set_state(pipeline,GST_STATE_NULL);
            gst_object_unref(sink);gst_object_unref(bus);gst_object_unref(pipeline);
            {std::lock_guard<std::mutex> l(frameMutex_);latest_=QImage();}
            lastDecodedMs_=0;

            bool report=false;
            {
                std::lock_guard<std::mutex> l(commandMutex_);
                if(!failure.isEmpty() && desired_ && generation_==gen){desired_=false;report=true;}
            }
            if(report)publishError(failure);
        }
    }

    void pollFrame(){
        QImage f;{std::lock_guard<std::mutex> l(frameMutex_);f.swap(latest_);}
        if(!f.isNull())emit frameReady(f);
    }
public:
    explicit GstRtspReceiver(QObject* p=nullptr):QObject(p){
        displayTimer_.setInterval(16);displayTimer_.setTimerType(Qt::PreciseTimer);
        connect(&displayTimer_,&QTimer::timeout,this,&GstRtspReceiver::pollFrame);
        displayTimer_.start();
        worker_=std::thread([this]{workerLoop();});
    }
    ~GstRtspReceiver()override{
        {
            std::lock_guard<std::mutex> l(commandMutex_);quit_=true;desired_=false;++generation_;
        }
        commandCv_.notify_all();
        if(worker_.joinable())worker_.join();
    }
    bool active()const{
        std::lock_guard<std::mutex> l(const_cast<std::mutex&>(commandMutex_));
        return desired_||starting_.load()||running_.load();
    }
    bool running()const{return running_.load();}
    void start(const QString& url){
        {
            std::lock_guard<std::mutex> l(commandMutex_);
            desiredUrl_=url;desired_=true;++generation_;
        }
        commandCv_.notify_all();
    }
    void stop(){
        {
            std::lock_guard<std::mutex> l(commandMutex_);desired_=false;++generation_;
        }
        commandCv_.notify_all();
        {std::lock_guard<std::mutex> l(frameMutex_);latest_=QImage();}
    }
signals:
    void frameReady(const QImage&);
    void streamError(const QString&);
};

class MainWindow:public QMainWindow{
    Q_OBJECT
    Config cfg_;
    QTcpSocket* sock_=nullptr;
    QByteArray rx_;
    QTimer reconnectTimer_,pingTimer_,videoRetry_;
    GstRtspReceiver* video_=nullptr;
    VideoView* view_=nullptr;
    QPushButton *detBtn_=nullptr,*aiBtn_=nullptr;
    QLabel *link_=nullptr,*videoState_=nullptr,*detState_=nullptr,*aiState_=nullptr,*target_=nullptr,*x_=nullptr,*y_=nullptr,*fps_=nullptr,*frame_=nullptr;
    bool det_=false,aiReq_=false,aiReady_=false,aiActive_=false,targetValid_=false,streamReady_=false;

    QString rtspUrl()const{return QString("rtsp://%1:%2/main.264").arg(cfg_.jetsonIp).arg(cfg_.rtspPort);}
    void sendLine(const QString& s){
        if(sock_->state()==QAbstractSocket::ConnectedState){sock_->write((s+"\n").toUtf8());sock_->flush();}
    }
    void setPill(QLabel* l,const QString& t,const char* state){l->setText(t);l->setProperty("state",state);l->style()->unpolish(l);l->style()->polish(l);l->update();}
    QLabel* pill(const QString& t){auto* l=new QLabel(t);l->setObjectName("pill");l->setAlignment(Qt::AlignCenter);l->setMinimumHeight(34);return l;}
    QLabel* metric(const QString& title,const QString& value,QVBoxLayout* parent){
        auto* box=new QFrame;box->setObjectName("metric");auto* l=new QVBoxLayout(box);auto* a=new QLabel(title);a->setObjectName("metricTitle");auto* b=new QLabel(value);b->setObjectName("metricValue");l->addWidget(a);l->addWidget(b);parent->addWidget(box);return b;
    }
    void startVideo(){
        if(!det_||!streamReady_||video_->active())return;
        setPill(videoState_,"CONNECTING","warn");
        view_->clearToMessage("CONNECTING TO LIVE DETECTION...");
        video_->start(rtspUrl());
    }
    void stopVideo(const QString& msg){
        videoRetry_.stop();video_->stop();view_->clearToMessage(msg);setPill(videoState_,det_?"WAIT":"OFF",det_?"warn":"neutral");
    }
    void videoError(const QString& why){
        video_->stop();
        if(!det_){
            view_->clearToMessage("DETECTION OFF\n\nNo video transmitted");
            setPill(videoState_,"OFF","neutral");
            return;
        }
        if(!streamReady_){
            view_->clearToMessage("DETECTION STARTING...\n\nWAITING FOR FIRST FINAL AIR FRAME");
            setPill(videoState_,"WAIT AIR","warn");
            return;
        }
        view_->clearToMessage("VIDEO LINK RECOVERING...\n\n"+why);
        setPill(videoState_,"RECOVERING","warn");
        videoRetry_.start(800);
    }
    void parseState(const QList<QByteArray>& p){
        if(p.size()<13)return;
        det_=p[2].toInt()!=0;
        aiReq_=p[3].toInt()!=0;
        aiReady_=p[4].toInt()!=0;
        aiActive_=p[5].toInt()!=0;
        targetValid_=p[6].toInt()!=0;
        const int xv=p[7].toInt(),yv=p[8].toInt();
        const double proc=p[9].toDouble(),src=p[10].toDouble();
        const auto fr=p[11].toULongLong();
        const QString st=QString::fromUtf8(p[12]);
        streamReady_=(p.size()>=14 && p[13].toInt()!=0);

        setPill(detState_,det_?"DETECTION ON":"DETECTION OFF",det_?"ok":"neutral");
        setPill(aiState_,aiActive_?"AI ON":(aiReq_?(aiReady_?"AI STARTING":"AI UNAVAILABLE"):"AI OFF"),aiActive_?"ok":(aiReq_?"warn":"neutral"));
        setPill(target_,targetValid_?"TARGET LOCKED":st,targetValid_?"ok":"warn");
        x_->setText(targetValid_?QString("%1%2 px").arg(xv>=0?"+":"").arg(xv):"-- px");
        y_->setText(targetValid_?QString("%1%2 px").arg(yv>=0?"+":"").arg(yv):"-- px");
        fps_->setText(QString("%1 / %2").arg(proc,0,'f',1).arg(src,0,'f',1));
        frame_->setText(QString::number(fr));
        detBtn_->setText(det_?"TURN DETECTION OFF":"TURN DETECTION ON");
        aiBtn_->setEnabled(det_&&sock_->state()==QAbstractSocket::ConnectedState);
        aiBtn_->setText(aiReq_?"TURN AI OFF":"TURN AI ON");

        if(!det_){
            videoRetry_.stop();
            if(video_->active()) video_->stop();
            view_->clearToMessage("DETECTION OFF\n\nNo video transmitted");
            setPill(videoState_,"OFF","neutral");
            return;
        }

        if(!streamReady_){
            videoRetry_.stop();
            if(video_->active()) video_->stop();
            view_->clearToMessage("DETECTION STARTING...\n\nWAITING FOR FIRST FINAL AIR FRAME");
            setPill(videoState_,"WAIT AIR","warn");
            return;
        }

        if(!video_->active()&&!videoRetry_.isActive()) startVideo();
    }
    void parseLine(QByteArray line){
        line=line.trimmed();if(line.isEmpty())return;const auto p=line.split(' ');
        if(p[0]=="HELLO"){setPill(link_,"CONTROL LINK LIVE","ok");detBtn_->setEnabled(true);sendLine("GET");}
        else if(p[0]=="STATE")parseState(p);
        else if(p[0]=="ACK")statusBar()->showMessage(QString::fromUtf8(line),1200);
        else if(p[0]=="ERR")statusBar()->showMessage(QString::fromUtf8(line),3000);
    }
public:
    explicit MainWindow(Config c):cfg_(std::move(c)){
        setWindowTitle("NEWSPACE GROUND - FRESH TCP R3");resize(1500,900);setMinimumSize(1100,700);
        auto* central=new QWidget;setCentralWidget(central);auto* root=new QHBoxLayout(central);root->setContentsMargins(16,16,16,16);root->setSpacing(16);
        view_=new VideoView;root->addWidget(view_,3);
        auto* side=new QFrame;side->setObjectName("side");side->setMinimumWidth(330);side->setMaximumWidth(420);auto* s=new QVBoxLayout(side);root->addWidget(side,1);
        auto* title=new QLabel("NEWSPACE VISION");title->setObjectName("title");s->addWidget(title);auto* sub=new QLabel("Fresh RTSP/TCP - readiness handshake R3");sub->setObjectName("subtitle");s->addWidget(sub);
        link_=pill("CONTROL LINK WAIT");videoState_=pill("VIDEO OFF");detState_=pill("DETECTION OFF");aiState_=pill("AI OFF");target_=pill("TARGET --");s->addWidget(link_);s->addWidget(videoState_);s->addWidget(detState_);s->addWidget(aiState_);s->addWidget(target_);
        detBtn_=new QPushButton("TURN DETECTION ON");aiBtn_=new QPushButton("TURN AI ON");detBtn_->setEnabled(false);aiBtn_->setEnabled(false);s->addWidget(detBtn_);s->addWidget(aiBtn_);
        x_=metric("X ERROR (+RIGHT)","-- px",s);y_=metric("Y ERROR (+UP)","-- px",s);fps_=metric("PROC / SOURCE FPS","0 / 0",s);frame_=metric("AIR FRAME","0",s);s->addStretch(1);
        auto* addr=new QLabel(QString("JETSON %1 | CTRL %2 | RTSP %3").arg(cfg_.jetsonIp).arg(cfg_.controlPort).arg(cfg_.rtspPort));addr->setWordWrap(true);addr->setObjectName("footer");s->addWidget(addr);
        setStyleSheet(R"(
            QMainWindow,QWidget{background:#0b1017;color:#e8eef7;font-family:Sans Serif;} #videoFrame{background:#020407;border:1px solid #273244;border-radius:10px;} #videoLabel{font-size:20px;color:#8c9bad;} #side{background:#111925;border:1px solid #273244;border-radius:12px;} #title{font-size:28px;font-weight:700;} #subtitle{color:#8c9bad;margin-bottom:10px;} #pill{background:#192433;border-radius:7px;padding:6px;font-weight:700;} #pill[state="ok"]{background:#153526;color:#83efb0;} #pill[state="warn"]{background:#463618;color:#ffd377;} #pill[state="neutral"]{background:#202a37;color:#a8b4c3;} QPushButton{min-height:46px;background:#2463eb;border:0;border-radius:8px;font-size:15px;font-weight:700;padding:8px;} QPushButton:hover{background:#3475ff;} QPushButton:disabled{background:#263343;color:#667486;} #metric{background:#0d141e;border:1px solid #253245;border-radius:8px;} #metricTitle{font-size:11px;color:#8fa0b5;} #metricValue{font-size:25px;font-weight:700;} #footer{color:#7f8da1;font-size:11px;}
        )");

        sock_=new QTcpSocket(this);video_=new GstRtspReceiver(this);
        connect(sock_,&QTcpSocket::connected,this,[this]{qInfo()<<"[CTRL] connected";setPill(link_,"CONTROL LINK LIVE","ok");detBtn_->setEnabled(true);sendLine("HELLO 1");});
        connect(sock_,&QTcpSocket::readyRead,this,[this]{rx_+=sock_->readAll();int n;while((n=rx_.indexOf('\n'))>=0){auto l=rx_.left(n);rx_.remove(0,n+1);parseLine(l);}});
        connect(sock_,&QTcpSocket::disconnected,this,[this]{qWarning()<<"[CTRL] disconnected";setPill(link_,"CONTROL LINK WAIT","warn");detBtn_->setEnabled(false);aiBtn_->setEnabled(false);reconnectTimer_.start(700);});
        connect(sock_,&QTcpSocket::errorOccurred,this,[this](QAbstractSocket::SocketError){if(sock_->state()!=QAbstractSocket::ConnectedState)reconnectTimer_.start(700);});
        reconnectTimer_.setSingleShot(true);connect(&reconnectTimer_,&QTimer::timeout,this,[this]{if(sock_->state()==QAbstractSocket::UnconnectedState)sock_->connectToHost(cfg_.jetsonIp,quint16(cfg_.controlPort));});
        pingTimer_.setInterval(1000);connect(&pingTimer_,&QTimer::timeout,this,[this]{if(sock_->state()==QAbstractSocket::ConnectedState)sendLine("PING");});pingTimer_.start();
        videoRetry_.setSingleShot(true);connect(&videoRetry_,&QTimer::timeout,this,[this]{if(det_&&streamReady_)startVideo();});
        connect(video_,&GstRtspReceiver::frameReady,this,[this](const QImage& f){view_->setFrame(f);setPill(videoState_,"VIDEO LIVE","ok");});
        connect(video_,&GstRtspReceiver::streamError,this,&MainWindow::videoError,Qt::QueuedConnection);
        connect(detBtn_,&QPushButton::clicked,this,[this]{if(sock_->state()!=QAbstractSocket::ConnectedState)return;sendLine(QString("DET %1").arg(det_?0:1));});
        connect(aiBtn_,&QPushButton::clicked,this,[this]{if(!det_||sock_->state()!=QAbstractSocket::ConnectedState)return;sendLine(QString("AI %1").arg(aiReq_?0:1));});
        sock_->connectToHost(cfg_.jetsonIp,quint16(cfg_.controlPort));
    }
protected:
    void closeEvent(QCloseEvent* e)override{video_->stop();sock_->disconnectFromHost();QMainWindow::closeEvent(e);}
};
}

#include "subscriber_node.moc"

int main(int argc,char** argv){
    QApplication app(argc,argv);Config cfg;
    try{if(!parseArgs(argc,argv,cfg))return 0;}catch(const std::exception& e){qCritical("%s",e.what());return 2;}
    MainWindow w(cfg);w.show();return app.exec();
}
