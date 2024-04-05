#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <time.h>

float** arrMalloc(int h , int w)//分配内存
{
    float ** arr;
    arr = (float **)malloc(h * sizeof(float*));
    for(int i = 0 ; i < h ; i ++)
    {
        arr[i] = (float *)malloc(w * sizeof(float));
    }
    return arr;
}

void freeArr(float **arr, int h) {
    //先释放内部内存，再释放整体内存
    for (int i = 0; i < h; i++) {
        free(arr[i]);
    }
    free(arr);
}

float** DataReading(FILE *fin, int h , int w)
{
    //分配内存
    float ** arr1;
    arr1 = arrMalloc(h,w);

    //读入并存储入数组
    for(int x = 0 ; x < h ; x++)
    {
        for(int y = 0 ; y < w ; y++)
        {
            fscanf(fin,"%f",&arr1[x][y]);
        }
    }
    return arr1;
}

float clOneBlock_opt(float** image, float** filter,const int h, const int w, 
                const int x, const int y, const int k1, const int k2 , const float padding)
{
        //SIMD优化
        //创建空向量，以及填充向量
        float32x4_t vec = vdupq_n_f32(0.0f);
        float32x4_t pad = vdupq_n_f32(padding);
        int i1 = x - k1/2;
        int i2 = y - k2/2;

        //将数据分为4份，作为向量进入内存同时计算结果
        for(int i = 0 ; i < k1 * k2 ; i+=4)
        {
            if(!(i1 + i/k1 < 0 | i1 + i%k1 < 0 | i1 + i/k1 >= h | i1 + i%k1 >= w))
            {
                float32x4_t image_v = vld1q_f32(&image[i1 + i/k1][i1 + i%k1]);
                float32x4_t filter_v = vld1q_f32(&filter[i/k1][i%k1]);
                vec = vmlaq_f32(vec, image_v , filter_v);
            }
            else 
            {
                float32x4_t filter_v = vld1q_f32(&filter[i/k1][i%k1]);
                vec = vmlaq_f32(vec, pad, filter_v);
            }
        }
        float sum[4] = {0};
        vst1q_f32(sum, vec);
        float32_t result = sum[0] + sum[1] + sum[2] + sum[3];

        //对余数的额外处理
        if(k1 * k2 % 4 == 0)
        {
            result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding;
        }
        else if(k1 * k2 % 4 == 1)
        {   
            if(x + k1/2 < h & y + k2/2 < w)
            {
                result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding
                        + image[x + k1/2][y + k2/2] * filter[k1 - 1][k2 - 1];
            }
            else 
            {
                result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding
                        + padding * filter[k1 - 1][k2 - 1];
            }
        }
        else if(k1 * k2 % 4 == 2)
        {
            if(x + k1/2 < h & y + k2/2 < w)
            {
                result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding
                        + image[x + k1/2][y + k2/2] * filter[k1 - 1][k2 - 1]
                        + image[x + k1/2][y + k2/2 - 1] * filter[k1 - 1][k2 - 1 - 1];
            }
            else if (y + k2/2 - 1 < w)
            {
                result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding
                        + image[x + k1/2][y + k2/2] * filter[k1 - 1][k2 - 1]
                        + padding * filter[k1 - 1][k2 - 1 - 1];
            }
            else{
                result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding
                        + padding * filter[k1 - 1][k2 - 1]
                        + padding * filter[k1 - 1][k2 - 1 - 1];               
            }
        }
        else
        {
            if(x + k1/2 < h & y + k2/2 < w)
            {
                result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding
                        + image[x + k1/2][y + k2/2] * filter[k1 - 1][k2 - 1]
                        + image[x + k1/2][y + k2/2 - 1] * filter[k1 - 1][k2 - 1 - 1]
                        + image[x + k1/2][y + k2/2 - 2] * filter[k1 - 1][k2 - 1 - 2];
            }
            else if (y + k2/2 - 1 < w)
            {
                result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding
                        + image[x + k1/2][y + k2/2] * filter[k1 - 1][k2 - 1]
                        + image[x + k1/2][y + k2/2 - 1] * filter[k1 - 1][k2 - 1 - 1]
                        + padding * filter[k1 - 1][k2 - 1 - 2];
            }
            else if (y + k2/2 - 2 < w){
                result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding
                        + image[x + k1/2][y + k2/2] * filter[k1 - 1][k2 - 1]
                        + padding * filter[k1 - 1][k2 - 1 - 1]
                        + padding * filter[k1 - 1][k2 - 1 - 2];             
            }
            else{
                result += (k1 * k2 - vaddvq_u32(vcltq_f32(vec, vdupq_n_f32(0.0f)))) * padding
                        + padding * filter[k1 - 1][k2 - 1]
                        + padding * filter[k1 - 1][k2 - 1 - 1]
                        + padding * filter[k1 - 1][k2 - 1 - 2];
            }
            
        }
        return result;
}

//实际上并未使用该方法，该方法在Convolutional中直接使用，而不是调用
float clOneBlock(float** image, float** filter,const int h, const int w, 
                const int x, const int y, const int k1, const int k2 , const float padding)
{
    float result = 0;
    int i1 = x - k1/2;
    int i2 = y - k2/2;
    for(int f1 = 0 ; f1 < k1 ; f1++)
    {
        for(int f2 = 0 ; f2 < k2 ; f2++)
        {
            if(!(f1 + i1 < 0 | f2 + i2 < 0 | f1 + i1 >= h | f2 + i2 >= w))
            {
                result += image[i1 + f1][i2 + f2] * filter[f1][f2];
            }
            else
            {
                result += padding * filter[f1][f2];
            }
        }
    }
    return result;
}

float** Convolutional(float** image,float** filter,const int h , const int w,
                    const int k1,const int k2 , const int padding, const int strides)
{
    float ** arr = arrMalloc(h,w);
    //对每一个结果像素点循环
    for(int x = 0 ; x < h ; x += strides)
    {
        for (int y = 0 ; y < w ; y+= strides)
        {
            //以下即为clOneBlock方法
            float result = 0;
            int i1 = x - k1/2;
            int i2 = y - k2/2;
            //对每一个卷积图像素循环
            for(int f1 = 0 ; f1 < k1 ; f1++)
            {
                for(int f2 = 0 ; f2 < k2 ; f2++)
                {
                    
                    if(!(f1 + i1 < 0 | f2 + i2 < 0 | f1 + i1 >= h | f2 + i2 >= w))//未超出边界
                    {
                        result += image[i1 + f1][i2 + f2] * filter[f1][f2];
                    }
                    else//超出边界，使用填充计算
                    {
                        result += padding * filter[f1][f2];
                    }
                }
            }
            arr[x][y] = result;
            //------------------------------
        }
    }
    return arr;
}

//仅为Convolutional但是替换clOneBlock为clOneBlock_opt
float** Convolutional_opt(float** image,float** filter,const int h , const int w,
                      const int k1,const int k2 , const int padding, const int strides)
{
    float ** arr = arrMalloc(h,w);
    for(int x = 0 ; x < h ; x += strides)
    {
        for (int y = 0 ; y < w ; y+= strides)
        {
            arr[x][y] = clOneBlock_opt(image , filter , h, w, x , y , k1 , k2 , padding);
        }
    }

    return arr;
}

int main(int argc, char* argv[])
{
    FILE *fin;//打开图片数据
    fin = fopen("untitled4.txt","r");
    FILE *fin2;//打开filter数据
    fin2 = fopen("filter5.txt","r");
    
    //读入高，宽，层数数据
    int h;
    int w;
    int layer;
    fscanf(fin,"%d",&h);
    fscanf(fin,"%d",&w);
    fscanf(fin,"%d",&layer);

    //分配RGB3层数组存储内存
    float ** arr1;
    float ** arr2;
    float ** arr3;
    arr1 = arrMalloc(h,w);
    arr2 = arrMalloc(h,w);
    arr3 = arrMalloc(h,w);

    //读入并存储入数组
    for(int x = 0 ; x < h ; x++)
    {
        for(int y = 0 ; y < w ; y++)
        {
            fscanf(fin,"%f",&arr1[x][y]);
            fscanf(fin,"%f",&arr2[x][y]);
            fscanf(fin,"%f",&arr3[x][y]);
        }
    }

    //对filter同样操作
    int h2;
    int w2;
    fscanf(fin2,"%d",&h2);
    fscanf(fin2,"%d",&w2);
    float ** filter = DataReading(fin2,h2,w2);

    //input kernal
    int kernal1 = h2;
    int kernal2 = w2;

    //intput padding 此处为手动在程序中更改
    int padding = 0;

    //input strides 此处为手动在程序中更改
    int strides = 1;

    clock_t start, end;//计时器

    start = clock();//开始计时
    //优化前程序
    float ** output1 = Convolutional(arr1, filter , h , w , kernal1 , kernal2 , padding , strides );
    float ** output2 = Convolutional(arr2 , filter , h , w , kernal1 , kernal2 , padding , strides );
    float ** output3 = Convolutional(arr3 , filter , h , w , kernal1 , kernal2 , padding , strides );

    end = clock();//结束计时

    printf("normal: %f ms\n",(float)(end-start)/CLOCKS_PER_SEC);//输出所用时间

    start = clock();//开始计时
    //优化后程序
    float ** output4 = Convolutional_opt(arr1, filter , h , w , kernal1 , kernal2 , padding , strides );
    float ** output5 = Convolutional_opt(arr2 , filter , h , w , kernal1 , kernal2 , padding , strides );
    float ** output6 = Convolutional_opt(arr3 , filter , h , w , kernal1 , kernal2 , padding , strides );

    end = clock();//结束计时

    printf("optmized: %f ms",(float)(end-start)/CLOCKS_PER_SEC);//输出所用时间

    //以下为输出数据进入txt文本
    FILE *fptr;

    fptr = fopen("out4.txt", "w"); 

    for(int i = 0 ; i < h ; i++)
    {
        for(int t = 0 ; t < w ; t++)
        {
            fprintf(fptr, "%f ", output1[i][t]);
            fprintf(fptr, "%f ", output2[i][t]);
            fprintf(fptr, "%f ", output3[i][t]);
            fprintf(fptr, "\n");
        }
    }
    fclose(fptr); 

    //使用方法释放内存
    freeArr(arr1,h);
    freeArr(arr2,h);
    freeArr(arr3,h);
    freeArr(output1,h);
    freeArr(output2,h);
    freeArr(output3,h);

    return 0;
}