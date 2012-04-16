#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <XnCppWrapper.h>
#include <iostream>

xn::Context context;
xn::DepthGenerator depthGen;
xn::ImageGenerator imageGen;
xn::UserGenerator userGen;
XnMapOutputMode outputMode;
const char* CONFIG_XML_PATH = "config.xml";
 
void openni_init(){
    context.InitFromXmlFile(CONFIG_XML_PATH);// OpenNIコンテキストの作成
    context.FindExistingNode(XN_NODE_TYPE_USER, userGen );// UserGenerator
    context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGen);// DepthGenerator
    context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGen);// ImageGenerator
    imageGen.GetMapOutputMode(outputMode); // 出力サイズの取得
    depthGen.GetAlternativeViewPointCap().SetViewPoint(imageGen);//RGBと深度画像のズレ補正
}
void getRedArea(cv::Mat hsv, cv::Mat redarea){
    // hsvをそれぞれのチャネルごとに分割。
	cv::Mat onechanels[3];
    cv::split(hsv, onechanels);
    // それぞれのチャネルごとに閾値を設定して二値化
    cv::Mat huemask,chromamask,brightnessmask, out;
    // 色相 170-180 (色相の最大値が180なのでそちら側は感知しない。)
    cv::threshold(onechanels[0], huemask, 170, 255, CV_THRESH_BINARY);
    // 彩度 60以上
    cv::threshold(onechanels[1], chromamask, 60, 255, CV_THRESH_BINARY);
    // 明度 80以上
    cv::threshold(onechanels[2], brightnessmask, 80, 255, CV_THRESH_BINARY);
    // 条件を満たした領域をoutに設定
    cv::bitwise_and(huemask, chromamask, out);
    cv::bitwise_and(brightnessmask, out, out);
 
    // 収縮と膨張でノイズな領域を削除。「重要な処理」
    cv::erode(out, out, cv::Mat());
    cv::dilate(out, out, cv::Mat());
 
    out.copyTo(redarea);
}
void mainloop(){
	std::string windowname = "image";
    cv::namedWindow(windowname, 0); // OpenCVウィンドウ作成
 
    // OpenNIからのデータ取得用
    xn::ImageMetaData imageMD;
    xn::SceneMetaData sceneMD;
    xn::DepthMetaData depthMD;
    // OpenNIからのデータ格納用
    cv::Mat image(outputMode.nYRes, outputMode.nXRes,CV_8UC3);
    cv::Mat user (outputMode.nYRes, outputMode.nXRes,CV_16UC1);
    cv::Mat depth(outputMode.nYRes, outputMode.nXRes,CV_16UC1);
    // 作業領域
    cv::Mat hsv     (outputMode.nYRes, outputMode.nXRes,CV_8UC3);
    cv::Mat userarea(outputMode.nYRes, outputMode.nXRes,CV_8UC1);
    cv::Mat redarea (outputMode.nYRes, outputMode.nXRes,CV_8UC1);
    cv::Mat tmp     (outputMode.nYRes, outputMode.nXRes,CV_8UC1);
    cv::Mat paint   (outputMode.nYRes, outputMode.nXRes,CV_8UC3);
    cv::Mat out     (outputMode.nYRes, outputMode.nXRes,CV_8UC3); // 表示用
     
    //cv::vector<cv::vector<cv::Point> > contours;// 輪郭情報
    int key; // 入力されたキー
    //描画領域を初期化
    paint.setTo(0); 
    while (1) {
        // kinectからの情報更新待ち
        context.WaitAndUpdateAll();
        // imageMD取得& cv::Matにコピー
        imageGen.GetMetaData(imageMD);
        xnOSMemCopy(image.data,imageMD.Data(),image.step * image.rows);
        // User情報を取得しcv::Matにコピー
        userGen.GetUserPixels(0, sceneMD);
        xnOSMemCopy(user.data, sceneMD.Data(), user.step * user.rows);
        // depthMD取得& depthをcv::Matにコピー
        depthGen.GetMetaData(depthMD);
        xnOSMemCopy(depth.data,depthMD.Data(), depth.step * depth.rows);
 
        /* RGB映像 */
        // rgbをbgrに変換(OpenCVで出力するため)
        cv::cvtColor(image, image, CV_RGB2BGR);
        // 赤い領域を取得
        cv::cvtColor(image, hsv, CV_BGR2HSV);

        getRedArea(hsv, redarea);
    //    /* ユーザ領域 */
    //    // 16UC1 ⇒ 8UC1 ⇒ 二値化 ⇒ 輪郭検出 = ユーザの輪郭抽出
    //    user.convertTo(userarea, CV_8UC1);
    //    cv::threshold(userarea, userarea, 0, 255, CV_THRESH_BINARY);
    //    cv::findContours(userarea, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        // 16UC1 ⇒ 8UC1 ⇒ 二値化(ユーザ領域＝白) 
        user.convertTo(userarea, CV_8UC1); // 輪郭検出でuserareaが破壊されるので再変換
        cv::threshold(userarea, userarea, 0, 255, CV_THRESH_BINARY);
         
        /* 画面描画 */
        tmp.setTo(0);   // 一時領域の初期化
        // RGB映像を出力用にコピー
        image.copyTo(out);
        // paint(描画領域)へ人が持ってる赤い領域をコピー
        image.copyTo(tmp, userarea); // 人領域
        tmp.copyTo(paint, redarea); // 赤い領域
 
        // 表示映像に描画情報を上書き
        paint.copyTo(out, paint);
         
        //// 映像に人間の輪郭情報を描画(検出されていることを表示する)
        //for(unsigned int i = 0; i < contours.size(); i++){
        //    cv::drawContours(out, contours, i, CV_RGB(0x00,0x80,0x00));
        //}
        // 画面に表示
        cv::imshow(windowname, out);
         
        key = cv::waitKey(30);
        if(key == 0x1b){// escape押下時は終了
            break;
        }
        else if(key == 'c'){// 描画領域を初期化する   
            paint.setTo(0);
        }
    }
}
int main(int argc, char* argv[])
{
    openni_init();

    mainloop();
    context.Release();
    return 0;
}
