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
    context.InitFromXmlFile(CONFIG_XML_PATH);// OpenNI�R���e�L�X�g�̍쐬
    context.FindExistingNode(XN_NODE_TYPE_USER, userGen );// UserGenerator
    context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGen);// DepthGenerator
    context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGen);// ImageGenerator
    imageGen.GetMapOutputMode(outputMode); // �o�̓T�C�Y�̎擾
    depthGen.GetAlternativeViewPointCap().SetViewPoint(imageGen);//RGB�Ɛ[�x�摜�̃Y���␳
}
void getRedArea(cv::Mat hsv, cv::Mat redarea){
    // hsv�����ꂼ��̃`���l�����Ƃɕ����B
	cv::Mat onechanels[3];
    cv::split(hsv, onechanels);
    // ���ꂼ��̃`���l�����Ƃ�臒l��ݒ肵�ē�l��
    cv::Mat huemask,chromamask,brightnessmask, out;
    // �F�� 170-180 (�F���̍ő�l��180�Ȃ̂ł����瑤�͊��m���Ȃ��B)
    cv::threshold(onechanels[0], huemask, 170, 255, CV_THRESH_BINARY);
    // �ʓx 60�ȏ�
    cv::threshold(onechanels[1], chromamask, 60, 255, CV_THRESH_BINARY);
    // ���x 80�ȏ�
    cv::threshold(onechanels[2], brightnessmask, 80, 255, CV_THRESH_BINARY);
    // �����𖞂������̈��out�ɐݒ�
    cv::bitwise_and(huemask, chromamask, out);
    cv::bitwise_and(brightnessmask, out, out);
 
    // ���k�Ɩc���Ńm�C�Y�ȗ̈���폜�B�u�d�v�ȏ����v
    cv::erode(out, out, cv::Mat());
    cv::dilate(out, out, cv::Mat());
 
    out.copyTo(redarea);
}
void mainloop(){
	std::string windowname = "image";
    cv::namedWindow(windowname, 0); // OpenCV�E�B���h�E�쐬
 
    // OpenNI����̃f�[�^�擾�p
    xn::ImageMetaData imageMD;
    xn::SceneMetaData sceneMD;
    xn::DepthMetaData depthMD;
    // OpenNI����̃f�[�^�i�[�p
    cv::Mat image(outputMode.nYRes, outputMode.nXRes,CV_8UC3);
    cv::Mat user (outputMode.nYRes, outputMode.nXRes,CV_16UC1);
    cv::Mat depth(outputMode.nYRes, outputMode.nXRes,CV_16UC1);
    // ��Ɨ̈�
    cv::Mat hsv     (outputMode.nYRes, outputMode.nXRes,CV_8UC3);
    cv::Mat userarea(outputMode.nYRes, outputMode.nXRes,CV_8UC1);
    cv::Mat redarea (outputMode.nYRes, outputMode.nXRes,CV_8UC1);
    cv::Mat tmp     (outputMode.nYRes, outputMode.nXRes,CV_8UC1);
    cv::Mat paint   (outputMode.nYRes, outputMode.nXRes,CV_8UC3);
    cv::Mat out     (outputMode.nYRes, outputMode.nXRes,CV_8UC3); // �\���p
     
    //cv::vector<cv::vector<cv::Point> > contours;// �֊s���
    int key; // ���͂��ꂽ�L�[
    //�`��̈��������
    paint.setTo(0); 
    while (1) {
        // kinect����̏��X�V�҂�
        context.WaitAndUpdateAll();
        // imageMD�擾& cv::Mat�ɃR�s�[
        imageGen.GetMetaData(imageMD);
        xnOSMemCopy(image.data,imageMD.Data(),image.step * image.rows);
        // User�����擾��cv::Mat�ɃR�s�[
        userGen.GetUserPixels(0, sceneMD);
        xnOSMemCopy(user.data, sceneMD.Data(), user.step * user.rows);
        // depthMD�擾& depth��cv::Mat�ɃR�s�[
        depthGen.GetMetaData(depthMD);
        xnOSMemCopy(depth.data,depthMD.Data(), depth.step * depth.rows);
 
        /* RGB�f�� */
        // rgb��bgr�ɕϊ�(OpenCV�ŏo�͂��邽��)
        cv::cvtColor(image, image, CV_RGB2BGR);
        // �Ԃ��̈���擾
        cv::cvtColor(image, hsv, CV_BGR2HSV);

        getRedArea(hsv, redarea);
    //    /* ���[�U�̈� */
    //    // 16UC1 �� 8UC1 �� ��l�� �� �֊s���o = ���[�U�̗֊s���o
    //    user.convertTo(userarea, CV_8UC1);
    //    cv::threshold(userarea, userarea, 0, 255, CV_THRESH_BINARY);
    //    cv::findContours(userarea, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        // 16UC1 �� 8UC1 �� ��l��(���[�U�̈恁��) 
        user.convertTo(userarea, CV_8UC1); // �֊s���o��userarea���j�󂳂��̂ōĕϊ�
        cv::threshold(userarea, userarea, 0, 255, CV_THRESH_BINARY);
         
        /* ��ʕ`�� */
        tmp.setTo(0);   // �ꎞ�̈�̏�����
        // RGB�f�����o�͗p�ɃR�s�[
        image.copyTo(out);
        // paint(�`��̈�)�֐l�������Ă�Ԃ��̈���R�s�[
        image.copyTo(tmp, userarea); // �l�̈�
        tmp.copyTo(paint, redarea); // �Ԃ��̈�
 
        // �\���f���ɕ`������㏑��
        paint.copyTo(out, paint);
         
        //// �f���ɐl�Ԃ̗֊s����`��(���o����Ă��邱�Ƃ�\������)
        //for(unsigned int i = 0; i < contours.size(); i++){
        //    cv::drawContours(out, contours, i, CV_RGB(0x00,0x80,0x00));
        //}
        // ��ʂɕ\��
        cv::imshow(windowname, out);
         
        key = cv::waitKey(30);
        if(key == 0x1b){// escape�������͏I��
            break;
        }
        else if(key == 'c'){// �`��̈������������   
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
