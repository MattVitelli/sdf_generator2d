#include <iostream>
#include <opencv2/opencv.hpp>
#define NOMINMAX
#include <Windows.h>

cv::Mat computeSDF(cv::Mat img)
{
	cv::Mat sdf = cv::Mat(img.rows, img.cols, CV_32F);
	int baseStride = 0;
	uint8_t* imgData = img.data;
	float* sdfData = (float*)sdf.data;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++, baseStride++)
		{
			bool isWhite = imgData[baseStride] > 0;
			float minDist = std::numeric_limits<float>::max();
			cv::Vec2f pt(x, y);
			int otherStride = 0;
			for (int i = 0; i < img.rows; i++)
			{
				for (int j = 0; j < img.cols; j++, otherStride++)
				{
					if (imgData[baseStride] != imgData[otherStride])
					{
						cv::Vec2f ptO(j, i);
						cv::Vec2f delta = (pt - ptO);
						minDist = std::min(minDist, sqrt(delta.dot(delta)));
					}
				}
			}
			sdfData[baseStride] = isWhite ? -minDist : minDist;
		}
	}

	return sdf;
}

cv::Mat computeSDFFast(cv::Mat img)
{
	cv::Mat sdf = cv::Mat(img.rows, img.cols, CV_32F);
	cv::Mat sdfClosest = cv::Mat(img.rows, img.cols, CV_32FC2);
	cv::Mat origPointMat = cv::Mat(img.rows, img.cols, CV_32FC2);

	uint8_t* imgData = img.data;
	float* sdfData = (float*)sdf.data;
	cv::Vec2f* sdfPoint = (cv::Vec2f*)sdfClosest.data;
	cv::Vec2f* origPoint = (cv::Vec2f*)origPointMat.data;
	float infDist = std::numeric_limits<float>::max();
	int width = img.cols;
	int height = img.rows;

	int baseStride = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++, baseStride++)
		{
			bool isWhite = imgData[baseStride] > 0;
			sdfData[baseStride] = infDist;
			sdfPoint[baseStride] = cv::Vec2f(-1, -1);
			origPoint[baseStride] = cv::Vec2f(x, y);
			if(x > 0 && y > 0 && x < width-1 && y < height - 1)
			{
				if (imgData[baseStride - 1] != imgData[baseStride] || imgData[baseStride + 1] != imgData[baseStride])
				{
					sdfData[baseStride] = 0.5;
					sdfPoint[baseStride] = origPoint[baseStride];
				}
				if (imgData[baseStride - width] != imgData[baseStride] || imgData[baseStride + width] != imgData[baseStride])
				{
					sdfData[baseStride] = 0.5;
					sdfPoint[baseStride] = origPoint[baseStride];
				}
			}	
		}
	}
	

	const float dx = 1;
	const float dy = 1;
	const float dxy = sqrt(2);

	baseStride = 0;
	for (int y = 1; y < height; y++)
	{
		int origBaseStride = y*width;
		baseStride = origBaseStride;
		for (int x = 0; x < width; x++, baseStride++)
		{
			if (x>0)
			{
				int cmpStride = baseStride - width - 1;
				if (sdfData[cmpStride] + dxy < sdfData[baseStride])
				{
					sdfPoint[baseStride] = sdfPoint[cmpStride];
					cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
					sdfData[baseStride] = sqrt(delta.dot(delta));
				}
			}

			{
				int cmpStride = baseStride - width;
				if (sdfData[cmpStride] + dy < sdfData[baseStride])
				{
					sdfPoint[baseStride] = sdfPoint[cmpStride];
					cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
					sdfData[baseStride] = sqrt(delta.dot(delta));
				}
			}

			if (x<width-1)
			{
				int cmpStride = baseStride - width + 1;
				if (sdfData[cmpStride] + dxy < sdfData[baseStride])
				{
					sdfPoint[baseStride] = sdfPoint[cmpStride];
					cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
					sdfData[baseStride] = sqrt(delta.dot(delta));
				}
			}
		}

		baseStride = origBaseStride + 1;
		for (int x = 1; x < width; x++, baseStride++)
		{
			int cmpStride = baseStride - 1;
			if (sdfData[cmpStride] + dx < sdfData[baseStride])
			{
				sdfPoint[baseStride] = sdfPoint[cmpStride];
				cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
				sdfData[baseStride] = sqrt(delta.dot(delta));
			}
		}

		baseStride = origBaseStride + width - 2;
		for (int x = width-2; x >= 0; x--, baseStride--)
		{
			int cmpStride = baseStride + 1;
			if (sdfData[cmpStride] + dx < sdfData[baseStride])
			{
				sdfPoint[baseStride] = sdfPoint[cmpStride];
				cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
				sdfData[baseStride] = sqrt(delta.dot(delta));
			}
		}
	}

	for (int y = height-2; y >= 0; y--)
	{
		int origBaseStride = y*width;
		baseStride = origBaseStride;
		for (int x = 0; x < width; x++, baseStride++)
		{
			if (x>0)
			{
				int cmpStride = baseStride + width - 1;
				if (sdfData[cmpStride] + dxy < sdfData[baseStride])
				{
					sdfPoint[baseStride] = sdfPoint[cmpStride];
					cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
					sdfData[baseStride] = sqrt(delta.dot(delta));
				}
			}

			{
				int cmpStride = baseStride + width;
				if (sdfData[cmpStride] + dy < sdfData[baseStride])
				{
					sdfPoint[baseStride] = sdfPoint[cmpStride];
					cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
					sdfData[baseStride] = sqrt(delta.dot(delta));
				}
			}

			if (x<width - 1)
			{
				int cmpStride = baseStride + width + 1;
				if (sdfData[cmpStride] + dxy < sdfData[baseStride])
				{
					sdfPoint[baseStride] = sdfPoint[cmpStride];
					cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
					sdfData[baseStride] = sqrt(delta.dot(delta));
				}
			}
		}

		baseStride = origBaseStride + 1;
		for (int x = 1; x < width; x++, baseStride++)
		{
			int cmpStride = baseStride - 1;
			if (sdfData[cmpStride] + dx < sdfData[baseStride])
			{
				sdfPoint[baseStride] = sdfPoint[cmpStride];
				cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
				sdfData[baseStride] = sqrt(delta.dot(delta));
			}
		}

		baseStride = origBaseStride + width - 2;
		for (int x = width - 2; x >= 0; x--, baseStride--)
		{
			int cmpStride = baseStride + 1;
			if (sdfData[cmpStride] + dx < sdfData[baseStride])
			{
				sdfPoint[baseStride] = sdfPoint[cmpStride];
				cv::Vec2f delta = sdfPoint[baseStride] - origPoint[baseStride];
				sdfData[baseStride] = sqrt(delta.dot(delta));
			}
		}
	}

	for (int stride = 0; stride < width*height; stride++)
	{
		if (imgData[stride] > 0)
		{
			sdfData[stride] = -sdfData[stride];
		}
	}

	return sdf;
}

float mean_error(cv::Mat img1, cv::Mat img2)
{
	assert(img1.rows == img2.rows && img1.cols == img2.cols);

	float* data1 = (float*)img1.data;
	float* data2 = (float*)img2.data;
	int totalElems = img1.rows*img1.cols;
	float error = 0.0f;
	for (int idx = 0; idx < totalElems; idx++)
	{
		error += std::abs(data1[idx] - data2[idx]);
	}
	return error / totalElems;
}

void run_timing_test(cv::Mat img)
{
	LARGE_INTEGER frequency, start, stop;
	QueryPerformanceFrequency(&frequency);
	int numRuns = 100;
	double netT = 0;
	for (int it = 0; it < numRuns; it++)
	{
		QueryPerformanceCounter(&start);
		cv::Mat result = computeSDFFast(img);
		QueryPerformanceCounter(&stop);
		double t = (double)(stop.QuadPart - start.QuadPart) / (double)frequency.QuadPart;
		netT += t;
	}
	netT /= (double)numRuns;
	std::cout << "Average time is " << netT << std::endl;
}

int main(int argc, char** argv)
{
	std::cout << "Hello, world" << std::endl;

	cv::Mat imgBig = cv::imread("img0.png");
	cv::cvtColor(imgBig, imgBig, CV_RGB2GRAY);
	assert(imgBig.channels() == 1);
	cv::Mat img;
	cv::resize(imgBig, img, cv::Size(256, 256));
	img = img > 0;
	//cv::Mat sdf = computeSDF(img); sdf /= img.rows;
	cv::Mat sdf_fast = computeSDFFast(img);	sdf_fast /= img.rows;
	run_timing_test(img);
	//std::cout << "Error is " << mean_error(sdf, sdf_fast) << std::endl;
	cv::imshow("Image", img);
	cv::imshow("Fast SDF", sdf_fast);
	//cv::imshow("True SDF", sdf);
	cv::waitKey();

	return 0;
}