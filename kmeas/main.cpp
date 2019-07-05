//
// Created by jzx on 19-7-4.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <kmeas.h>


using namespace cv;
using namespace std;

/*
 *(1)kmeans算法的2个缺点：第一是必须人为指定所聚的类的个数k；
 * 第二是如果使用欧式距离来衡量相似度的话，可能会得到错误的结果，
 * 因为没有考虑到属性的重要性和相关性。为了减少这种错误，在使用kmeans距离时，
 * 一定要使样本的每一维数据归一化，不然的话由于样本的属性范围不同会导致错误的结果。
 * (2)kmeas_demo:
 * (3)kmeas:
 * (4)kmeas:
 * */
int kmeas_demo() {
    //fake points
    Mat img(500, 500, CV_8UC3);
    RNG rng(12345);
    Scalar colorTab[] = {
            Scalar(0, 0, 255),
            Scalar(255, 0, 0),
    };
    int numCluster = 2;
    int sampleCount = rng.uniform(5, 500);
    Mat points(sampleCount, 1, CV_32FC2);

    for (int k = 0; k < numCluster; k++) {
        Point center;
        center.x = rng.uniform(0, img.cols);
        center.y = rng.uniform(0, img.rows);
        Mat pointChunk = points.rowRange(k*sampleCount / numCluster,
                                         k == numCluster - 1 ? sampleCount : (k + 1)*sampleCount / numCluster);
        rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
    }
    randShuffle(points, 1, &rng);

    //
    Mat labels;
    Mat centers;
    kmeans(points, numCluster, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1), 3, KMEANS_PP_CENTERS, centers);

    // �ò�ͬ��ɫ��ʾ����
    img = Scalar::all(255);
    for (int i = 0; i < sampleCount; i++) {
        int index = labels.at<int>(i);
        Point p = points.at<Point2f>(i);
        circle(img, p, 2, colorTab[index], -1, 8);
    }

    // ÿ������������������Բ
    for (int i = 0; i < centers.rows; i++) {
        int x = centers.at<float>(i, 0);
        int y = centers.at<float>(i, 1);
        printf("c.x= %d, c.y=%d", x, y);
        circle(img, Point(x, y), 40, colorTab[i], 1, LINE_AA);
    }

    imshow("KMeans-Data-Demo", img);
    waitKey(0);
    return 0;
}

int kmeas_color() {
    Mat src = imread("/home/jzx/CLionProjects/k-means/1.jpg");
    if (src.empty()) {
        printf("could not load image...\n");
        return -1;
    }
    Mat dst;
    cv::resize(src, dst, cv::Size(src.cols * 0.25,src.rows * 0.25), 0, 0, CV_INTER_LINEAR);
    namedWindow("input image", WINDOW_AUTOSIZE);
    imshow("input image", dst);

    int width = dst.cols;
    int height = dst.rows;
    int dims = dst.channels();

    //初始化
    int sampleCount = width*height;
    int clusterCount = 2;
    Mat labels;
    Mat centers;

    // RGB 数据转换到样本数据
    Mat sample_data = dst.reshape(3, sampleCount);
    Mat data;
    sample_data.convertTo(data, CV_32F);

    // K-Means
    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
    kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

    Mat card = Mat::zeros(Size(width, 50), CV_8UC3);
    vector<float> clusters(clusterCount);
    for (int i = 0; i < labels.rows; i++) {
        clusters[labels.at<int>(i, 0)]++;
    }
    for (int i = 0; i < clusters.size(); i++) {
        clusters[i] = clusters[i] / sampleCount;
    }
    int x_offset = 0;
    for (int x = 0; x < clusterCount; x++) {
        Rect rect;
        rect.x = x_offset;
        rect.y = 0;
        rect.height = 50;
        rect.width = round(clusters[x] * width);
        x_offset += rect.width;
        int b = centers.at<float>(x, 0);
        int g = centers.at<float>(x, 1);
        int r = centers.at<float>(x, 2);
        rectangle(card, rect, Scalar(b, g, r), -1, 8, 0);
    }

    imshow("Image Color Card", card);
    waitKey(0);
    return 0;
}

int kmeas_segmentation() {
    Mat src = imread("/home/jzx/CLionProjects/k-means/5.jpg");
    if (src.empty()) {
        printf("could not load image...\n");
        return -1;
    }
    Scalar colorTab[] = {
            Scalar(0, 0, 255),
            Scalar(0, 255, 0),
            Scalar(255, 0, 0),
            Scalar(0, 255, 255),
            Scalar(255, 0, 255)
    };
    cv::resize(src, src, cv::Size(src.cols * 0.25,src.rows * 0.25), 0, 0, CV_INTER_LINEAR);
    namedWindow("input image", WINDOW_AUTOSIZE);
    imshow("input image", src);
    int width = src.cols;
    int height = src.rows;
    int dims = src.channels();
    // 初始化
    int sampleCount = width*height;
    int clusterCount = 2;
    Mat labels;
    Mat centers;

    // 传数据
    Mat sample_data = src.reshape(3, sampleCount);
    Mat data;
    sample_data.convertTo(data, CV_32F);

    // K-Means
    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
    kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);
    // 画
    int index = 0;
    Mat result = Mat::zeros(src.size(), src.type());
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            index = row*width + col;
            std::cout<<index<<std::endl;
            int label = labels.at<int>(index, 0);
            result.at<Vec3b>(row, col)[0] = colorTab[label][0];
            result.at<Vec3b>(row, col)[1] = colorTab[label][1];
            result.at<Vec3b>(row, col)[2] = colorTab[label][2];
        }
    }

    imshow("KMeans-image", result);
    waitKey(0);
    return 0;
}
int main(int argc, char** argv){
    kmeas_demo();
    //kmeas_color();
    //kmeas_segmentation();
    return 1;
}