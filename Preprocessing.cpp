#include <string>
#include <opencv2/opencv.hpp>
#include "utils/utils.hpp"

#pragma warning(disable:4996)
#define _CRT_SECURE_NO_WARNINGS

using namespace std;

bool mjroi(const cv::Mat image, vector<cv::Point2f>& corners, cv::Mat& _out = cv::Mat());

bool mjcards(const cv::Mat image, const vector<cv::Point2f>& corners, cv::Mat& out, vector<float>& boundary);

const float k_min = 0.712389f;
const float k_thre = 0.843367f;
const float k_max = 0.878161f;

#include <stdio.h>
#include <time.h>

int tin = 0;
int hu = 0;
int none = 0;

void eval(void);
int K = -1;     /// how many cards in an image, for determinging play mode
int main(int argc, char** argv)
{
    //eval();

    //std::cout << "tin: " << tin << endl;
    //std::cout << "hu: " << hu << endl;
    //std::cout << "none: " << none << endl;

    //system("pause");
    //return 0;

    const cv::String keys =
        "{help h usage ? |      | print this message }"
        "{@image         |      | input image }"
        "{@output        |      | path to output file }"
        "{cards k        |-1    | number of cards (default: automatically determine) }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Application name v1.0.0");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    K = parser.get<int>("cards");
    if (K != 13 || K != 14)
        K = -1;

    cv::String imgFile = parser.get<cv::String>(0);
    cv::String outFile = parser.get<cv::String>(1);

    if (imgFile.empty() || outFile.empty() || !parser.check())
    {
        parser.printErrors();
        return 0;
    }

    cv::Mat im = cv::imread(imgFile);
    if (im.empty())
    {
        std::cout << "Cannot open image file: " << imgFile << endl;
        return 0;
    }

    if (!DirectoryExists("tmp")) {
        system("mkdir tmp");
    }

    FILE* fp = NULL;
    if (!(fp = fopen(outFile.c_str(), "w")) && !(fp = fopen("./tmp/default.txt", "w")))
    {
        return -1;
    }

    if (im.cols > 960 || im.rows > 540)
    {
        int x1 = (std::max)((im.cols - 960) / 2, 0);
        int y1 = (std::max)((im.rows - 540) / 2, 0);
        int x2 = (std::min)(im.cols - x1, im.cols);
        int y2 = (std::min)(im.rows - y1, im.rows);

        im = im(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
    }
    //imwrite("./tmp/default.jpg", im);

    /// detect the roi of mj cards
    vector<cv::Point2f> corners;
    if (mjroi(im, corners))
    {
        ///////// detect the bounding box the mj cards
        cv::Mat out;
        vector<float> cards;
        if (mjcards(im, corners, out, cards) && !out.empty())
        {
            int tick = cv::getTickCount() % (100000 * K) * 10;

            fprintf(fp, "%d\n", K);
            for (int i = 0; i < K; i++)
            {
                cv::Mat card_img;

                cv::resize(out(cv::Rect(cv::Point((int)(cards[i] * out.cols), 0), 
                                cv::Point((int)(cards[i + 1] * out.cols), out.rows))),
                            card_img, cv::Size(256, 256));

                string name = cv::format("./tmp/%08d.jpg", tick + i);
                cv::imwrite(name, card_img);
                fprintf(fp, "%s\n", name.c_str());
            }
            fclose(fp);
            return 1;
        }
    }

    fprintf(fp, "0\n");
    fclose(fp);
    return 0;
}


cv::Mat _tmp;
void eval(void)
{
    string infolder("./bad/"), outfolder("./tmp/");

    //string infolder("./suo14_te/"), outfolder("./tmp/");
    //string infolder("./suo13_te/"), outfolder("./tmp/");
    //string infolder("./data/cards13/"), outfolder("./tmp/");
    //string infolder("./data/cards14/"), outfolder("./tmp/");

    //string infolder("./data/hu_no/"), outfolder("./tmp/");
    //string infolder("./data/hu_yes/"), outfolder("./tmp/");

    //string infolder("./data/tin_0/"), outfolder("./tmp/");
    //string infolder("./data/tin_8/"), outfolder("./tmp/");
    //string infolder("./data/tin_9/"), outfolder("./tmp/");
    //string infolder("./data/tin_25/"), outfolder("./tmp/");
    //string infolder("./data/tin_36/"), outfolder("./tmp/");
    //string infolder("./data/tin_147/"), outfolder("./tmp/");
    //string infolder("./data/tin_2578/"), outfolder("./tmp/");
    //string infolder("./data/tin_34567/"), outfolder("./tmp/");
    //string infolder("./data/tin_56789/"), outfolder("./tmp/");

    vector<string> filenames;
    get_files_in_directory(infolder, filenames);

    vector<string>::iterator it, it_end;
    for (it = filenames.begin(), it_end = filenames.end(); it != it_end; it++)
    {
        cv::Mat im = cv::imread(infolder + SEP + *it);

        cv::Mat out;
        /// detect the roi of mj cards
        vector<cv::Point2f> corners;
        if (!mjroi(im, corners, out))
        {
            continue;
        }

        //if (!out.empty())
        //{
        //    for (int i = 0; i < (int)corners.size(); i++)
        //        cv::line(out, corners[i], corners[(i+1)% (int)corners.size()], 
        //                    cv::Scalar(255, 0, 0), 2);

        //    for (int i = 0; i < corners.size(); i++)
        //        cv::circle(out, corners[i], 1, cv::Scalar(255, 255, 0), 2);

        //    cv::imwrite(outfolder + SEP + *it, out);
        //}

        //continue;

        ///////// detect the bounding box the mj cards
        //cv::Mat out;
        out = cv::Mat();
        vector<float> cards;
        K = -1;
        if (mjcards(im, corners, out, cards) && !out.empty())
        {
            //cv::resize(out, out, cv::Size(), 3.0, 3.0);

            for (int i = 0; i < (int)cards.size(); i++)
                cv::line(out, cv::Point((int)(cards[i] * out.cols), 0), cv::Point((int)(cards[i] * out.cols), out.rows), cv::Scalar(255, 0, 0), 2);

            cv::imwrite(outfolder + SEP + *it, out);
        }
        else
            std::cout << "is a bad image: " << *it << endl;
    }

    //for (int i = 0; i < (int) card_ratios.size(); i++)
    //std::cout << card_ratios[i] << std::endl;

    //system("pause");
    return;
}


bool mjcards(const cv::Mat _img, const vector<cv::Point2f>& _corners, cv::Mat& _out, vector<float>& _cards)
{
    if (_corners.size() != 4)
    {
        std::cout << "# corners = " << _corners.size() << endl;
        return false;
    }

    /// warp the image
    float roi_w = ((float)cv::norm(_corners[0] - _corners[1]) + (float)cv::norm(_corners[2] - _corners[3])) / 2.0f;
    float roi_h = ((float)cv::norm(_corners[0] - _corners[3]) + (float)cv::norm(_corners[2] - _corners[1])) / 2.0f;
    cv::Point2f roi_c = (_corners[0] + _corners[1] + _corners[2] + _corners[3]) / 4.0f;

    vector< cv::Point2f> target(4, roi_c);
    target[0] += cv::Point2f(-roi_w/2.0f, -roi_h/2.0f);
    target[1] += cv::Point2f(+roi_w/2.0f, -roi_h/2.0f);
    target[2] += cv::Point2f(+roi_w/2.0f, +roi_h/2.0f);
    target[3] += cv::Point2f(-roi_w/2.0f, +roi_h/2.0f);

    cv::Rect roi = cv::Rect(cv::Point(target[0]), cv::Point(target[2]));
    if ((roi & cv::Rect(cv::Point(0, 0), cv::Point(_img.cols, _img.rows))) != roi)
    {
        std::cout << "did not find a proper roi: " << std::endl << cv::Mat(target) << endl;
        return false;
    }

    cv::Mat trans = cv::getPerspectiveTransform(_corners, target);
    cv::Mat tmp_im;
    cv::warpPerspective(_img, tmp_im, trans, _img.size());

    // extend the upper and bottom boundaries
    target[0].y = (std::max)(target[0].y - roi_h / 2.5f, 0.0f);
    target[1].y = (std::max)(target[1].y - roi_h / 2.5f, 0.0f);
    target[2].y = (std::min)(target[2].y + roi_h / 4.0f, (float)tmp_im.rows);
    target[3].y = (std::min)(target[3].y + roi_h / 4.0f, (float)tmp_im.rows);

    cv::Mat roi_im;
    cv::resize(tmp_im(cv::Rect(target[0], target[2])), roi_im, cv::Size(720, 120));

    //// select the middle section of the roi and convert it to HSV
    cv::Mat hsv;
    cv::cvtColor(roi_im(cv::Rect(cv::Point(0, 40), cv::Point(720, 80))), hsv, cv::COLOR_BGR2HSV);

    //// threshold the HSV image, keep only the red pixels
    cv::Mat lower_red_hue_range;
    cv::Mat upper_red_hue_range;
    cv::inRange(hsv, cv::Scalar(0, 50, 50), cv::Scalar(10, 255, 255), lower_red_hue_range);
    cv::inRange(hsv, cv::Scalar(170, 50, 50), cv::Scalar(180, 255, 255), upper_red_hue_range);

    //// combine the above two images
    cv::Mat red_hue_image;
    cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);

    int w = red_hue_image.cols;
    int h = red_hue_image.rows;

    int lt = w / 30, rt = w - w / 30;
    int lb = w / 30, rb = w - w / 30;

    int c = 128;
    unsigned char *pt = (unsigned char*)red_hue_image.row(0).data;
    unsigned char *pb = (unsigned char*)red_hue_image.row(h - 1).data;
    for (int count = 0; lt < w; lt++)
    {
        if (pt[lt] < c) count++;
        else count = 0;
        if (count > 3) { lt = lt - count; break; }
    }

    for (int count = 0; rt >= 0; rt--)
    {
        if (pt[rt] < c) count++;
        else count = 0;

        if (count > 3) { rt = rt + count; break; }
    }

    for (int count = 0; lb < w; lb++)
    {
        if (pb[lb] < c) count++;
        else count = 0;

        if (count > 3) { lb = lb - count; break; }
    }

    for (int count = 0; rb >= 0; rb--)
    {
        if (pb[rb] < c) count++;
        else count = 0;

        if (count > 3) { rb = rb + count; break; }
    }

    if (K < 0)
    {
        float r = 1.0f - float ((lt + lb + 2 * w - rt - rb) * h / 2) / (float)(w * h);
        //card_ratios.push_back(r);

        if (r > k_min)
        {
            if (r < k_thre)
                K = 13;

            else if (r < k_max)
                K = 14;
        }
    }

    if ((K+1) / 2 != 7)
    {
        cout << "The number of cards is not acceptable." << endl;

        none++;
        return false;
    }
    else if (K == 13)
        tin++;
    else
        hu++;


    const int card_width = 30;
    const int card_height = 45;
    /// do warping again
    vector<cv::Point2f> corners(4);
    corners[0] = cv::Point2f((float)lt, 40.0f);
    corners[1] = cv::Point2f((float)rt, 40.0f);
    corners[2] = cv::Point2f((float)rb, 80.0f);
    corners[3] = cv::Point2f((float)lb, 80.0f);

    float zoom = 3.0f;
    target[0] = cv::Point2f(0.0f, (float)card_height / 3.0f) * zoom;
    target[1] = cv::Point2f((float)(K*card_width), (float)card_height / 3.0f) * zoom;
    target[2] = cv::Point2f((float)(K*card_width), (float)card_height * 2.0f / 3.0f) * zoom;
    target[3] = cv::Point2f(0.0f, (float)card_height * 2.0f / 3.0f) * zoom;

    trans = cv::getPerspectiveTransform(corners, target);
    cv::warpPerspective(roi_im, _out, trans, cv::Size((int)(K * card_width * zoom), (int)(card_height * zoom)));

    //tmp_im = _out(cv::Rect(target[0], target[2])).clone();
    cv::resize(_out(cv::Rect(target[0], target[2])), tmp_im, cv::Size(), 1.0f/zoom, 1.0f / zoom);
    cv::medianBlur(tmp_im, tmp_im, 7);
    //cv::cvtColor(tmp_im, tmp_im, CV_BGR2GRAY);

    cv::Mat bgr[3];
    cv::split(tmp_im, bgr);                  //split src into B,G,R channels
    (cv::min)(bgr[0], bgr[1], tmp_im);       //find minimum between B and G channels
    (cv::min)(tmp_im, bgr[2], tmp_im);       //find minimum between temp and R channels

    cv::boxFilter(tmp_im, tmp_im, -1, cv::Size(11, 11));

    cv::Mat avg_im;
    cv::reduce(tmp_im, avg_im, 0, CV_REDUCE_AVG);
    unsigned char *p = (unsigned char *) avg_im.data;

    /// refine the boundary of the cards
    vector<int> index(K + 1);
    vector<int> cards(K+1);
    for(int i = 0; i <= K; i++)
    {
        index[i] = i;
        cards[i] = i * card_width;
        //if (i != 0 && i != K)
            //cv::line(_out, cv::Point(cards[i], 0), cv::Point(cards[i], card_height), cv::Scalar(0, 0, 255), 1);
    }

    int max_iter = 7;
    while (max_iter-- > 0)
    {
        random_shuffle(index.begin(), index.end());

        vector<int>::iterator it = index.begin(), it_end = index.end();
        for ( ; it != it_end; it++)
        {
            if ((*it)%K == 0 )
                continue;

            const int l = cards[(*it) + 1];
            const int r = cards[(*it) - 1];
            int best_b = cards[(*it)];

            double w1 = 1.2, w2 = 1.0;
            double max_val = p[best_b] * w1 - abs(2 * best_b - (l + r)) * w2;
            for (int j = -7; j < 8; j++)
            {
                if (j == 0)
                    continue;

                int b = cards[(*it)] + j;
                double v = p[b] * w1 - abs(2 * b - (l + r)) * w2;
                if (v > max_val)
                {
                    max_val = v;
                    best_b = b;
                }
            }

            cards[(*it)] = best_b;
        }
    }

    _cards.push_back(0.0f);
    for (int i = 1; i < K; i++)
        _cards.push_back((float)cards[i] / (float)(card_width * K));
    _cards.push_back(1.0f);

    return true;
}



bool mjroi(const cv::Mat _image, vector<cv::Point2f>& _corners, cv::Mat& _tmp)
{
    float scale = 0.4f;
    cv::Mat bgr, hsv;
    cv::resize(_image, bgr, cv::Size(), scale, scale);

    // Convert input image to HSV
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // Threshold the HSV image, keep only the red pixels
    cv::Mat lower_red_hue_range;
    cv::Mat upper_red_hue_range;

    cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(7, 255, 255), lower_red_hue_range);
    cv::inRange(hsv, cv::Scalar(170, 70, 50), cv::Scalar(180, 255, 255), upper_red_hue_range);

    // Combine the above two images
    cv::Mat red_hue_image;
    cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);

    vector< vector< cv::Point > > contours;
    cv::findContours(red_hue_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    int w = red_hue_image.cols;
    int h = red_hue_image.rows;

    const int minPerimeterPixels = (std::min)(w, h) / 5;
    const int maxPerimeterPixels = (std::max)(w, h) * 5;
    const double accuracyRate = 0.01;

    vector< cv::Point > candidates;
    for (unsigned int i = 0; i < contours.size(); i++) {

        // check perimeter
        int np = (int) contours[i].size();
        if (np < minPerimeterPixels || np > maxPerimeterPixels)
        {
            continue;
        }

        vector< cv::Point > approxCurve;
        approxPolyDP(contours[i], approxCurve, (double)np * accuracyRate, true);
        int nv = (int) approxCurve.size();
        if (nv < 4)
        {
            continue;
        }

        for (int j = 0; j < nv; j++) {
            candidates.push_back(approxCurve[j]);
        }
    }

    int m = (int) candidates.size();

    if (m < 4)
    {
        cout << "# of candidate points: " << m << endl;
        return false;
    }

    cv::Rect bbox = cv::boundingRect(candidates);
    vector< cv::Point > corners(4);
    //corners[0] = cv::Point(0, 0);
    //corners[1] = cv::Point(w, 0);
    //corners[2] = cv::Point(w, h);
    //corners[3] = cv::Point(0, h);
    corners[0] = bbox.tl();
    corners[1] = cv::Point(bbox.x + bbox.width, bbox.y);
    corners[2] = bbox.br();
    corners[3] = cv::Point(bbox.x, bbox.y + bbox.height);;

    for (int i = 0; i < 4; i++)
    {
        int index = 0;
        int min_dist = w * w + h * h;

        for (int j = 0; j < m; j++)
        {
            cv::Point p = candidates[j] - corners[i];
            int d = p.x * p.x + p.y * p.y;
            if (d < min_dist)
            {
                min_dist = d;
                index = j;
            }
        }
        _corners.push_back(cv::Point2f((float)candidates[index].x / scale, (float)candidates[index].y / scale));
    }

    return true;
}


const double suo13[] = { 0.80367826, 0.007749545, 0.720139, 0.830556 };
const double suo14[] = { 0.860101883, 0.003161334, 0.85, 0.875 };

int _main(int argc, char** argv)
{
    double mu13 = suo13[0], mu14 = suo14[0];
    double sigma13 = suo13[1], sigma14 = suo14[1];
    double var13 = sigma13 * sigma13, var14 = sigma14 * sigma14;

    double a = 1.0 / (2.0 * var13) - 1.0 / (2.0 * var14);
    double b = mu14 / var14 - mu13 / var13;
    double c = (mu13 * mu13) / (2.0 * var13) - (mu14 * mu14) / (2 * var14) - log(suo14[1] / suo13[1]);
    double D = b * b - 4.0 * a * c;
    double x1 = (-b + sqrt(D)) / (2.0 * a);
    double x2 = (-b - sqrt(D)) / (2.0 * a);

    std::cout << "k_min: " << suo13[2] - sigma13 << std::endl;
    if (x1 > mu13 && x1 < mu14)
        std::cout << "k_thre:  "<< x1 << std::endl;
    if (x2 > mu13 && x2 < mu14)
        std::cout << "k_thre: " << x2 << std::endl;
    std::cout << "k_max: " << suo14[3] + sigma14 << std::endl;

    system("pause");
    return 1;
}
