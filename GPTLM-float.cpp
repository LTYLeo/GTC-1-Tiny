 #include<bits/stdc++.h>
#include<windows.h>
#include<cmath>
#include<random>
#define MAXN 6
#define ALL 3
using namespace std;
float w[ALL][MAXN]={
{-0.24307,-0.83282,-0.09142,-0.75026,-0.7553},//-0.09447,0.7467,-0.8478,-0.4659,-0.0172,0.1566,-0.5676,0.7467,-0.8478,-0.4659,-0.0172,0.1566,-0.5676,0.7635,0.8384,0.1254,0.7635,-0.2973,-0.9725},
{-0.29236,0.50816,0.89024,0.02185,-0.68497,0.47142},//-0.01826,-0.30716,-0.05628,0.22472,0.2712,0.5851,0.7635,0.8384,0.1254,0.7635,-0.2973,-0.9725,0.4523,0.0024,-0.9194,0.2789,0.1354,-0.5263},
{-0.9322,-0.01826,-0.30716,-0.05628,0.22472,0.2712}//,0.5851,0.1245,-0.7596,-0.9194,0.2789,0.1354,-0.5263,0.2635,-0.4234,0.1333,0.4934,-0.2398,0.2345,0.6534,0.8542,-0.1233,0.7635,0.8384}
};
float x[ALL][MAXN]={
    {0.2345,0.6534,0.8542,-0.1233,0.7635,0.8384},//0.1254,0.7635,-0.2973,-0.9725,0.4523,0.0024,0.7467,-0.8478,-0.4659,-0.0172,0.1566,-0.5676,-0.9322,-0.01826,-0.30716,-0.05628,0.22472,0.2712},
    {-0.5263,0.2635,-0.4234,0.1333,0.4934,-0.2398},//0.2345,0.6534,0.8542,-0.1233,0.7635,0.8384,-0.9322,-0.01826,-0.30716,-0.05628,0.22472,0.2712,0.5851,0.1245,-0.7596,-0.9194,0.2789,0.1354},
    {0.2134,-0.4568,0.3248,0.2189,-0.9742,-0.3421}//,-0.01826,-0.30716,-0.05628,0.22472,0.2712,0.5851,0.2345,0.6534,0.8542,-0.1233,0.7635,0.8384,0.1254,0.7635,-0.2973,-0.9725,0.4523,0.0024}
};
string wl[ALL]={"1","2","3"};
int trainnum=0,max_trainnum=100; 
float f[MAXN];
float h[MAXN];
int used[MAXN];
float pro[ALL];
int cnt=-1;
int reall;
int pos;
float loss,minloss=1e9,preloss;
float delta;
float alpha = 0.1;
float minAlpha = 0.001;
float alphaDecayRate = 0.001;
float present[MAXN]; 
int fv[MAXN],fcnt=-1;
int cols=MAXN; 

float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}
void feature()
{
    for(int i=0;i<MAXN;i++)
    {
        f[i] = 0.0;
        for(int j=0;j<=cnt;j++)
        {
            f[i] += w[used[j]][i] * x[used[j]][i];
        }
    }
}

void toh()
{
    for(int i=0;i<MAXN;i++)
    {
        h[i] = sigmoid(f[i]);
    }
}

void softmax()
{
    float sum_exp=0;
    for(int i=0;i<ALL;i++)
    {
        float dot_product=0;
        for(int j=0;j<MAXN;j++)
        {
            dot_product += h[j] * x[i][j];
        }
        pro[i] = exp(dot_product);
        sum_exp += pro[i];
    }
    for(int i=0;i<ALL;i++)
    {
        pro[i] /= sum_exp;
    }
}
/*void fitting()
{
	loss = pro[pos] - pro[reall];
	//delta = loss*alpha;
	float maxx=0;
	default_random_engine e(time(0));
	uniform_real_distribution<double> u(0,1);
	while(1){
		for(int i=0;i<MAXN;i++){
			present[i]=w[used[cnt]][i];
		}
		int count=0;
		float temp_loss;
		srand((unsigned int)time(NULL));
		int random=rand()%49+0;
		delta=u(e);
		float temp=w[used[cnt]][random];
		temp+=delta;
		if(temp>1||temp<-1) temp/=10;
		w[used[cnt]][random]=temp;
		feature();
		toh();
		softmax();
		maxx=0;
	    for(int i=0; i<ALL; i++)
	    {
	    	if(pro[i] > maxx)
	        {
	            maxx = pro[i];
	            pos = i;
	       	}
	    }
	    temp_loss=pro[pos]-pro[reall];
	    if(temp_loss<=0){
	    	return;
		}
		alpha -= alphaDecayRate;
        if (alpha < minAlpha) {
            alpha = minAlpha;
        }
        //delta = loss*alpha;
		if(minloss>loss){
			minloss=loss;
		}
		if(minloss<=0){
			return;
		}
		cout<<"Trainnum:"<<trainnum << " Loss:" << minloss << " Delta:" << delta << endl;
		cout<<endl;
		for(int i=0;i<MAXN;i++) cout<<w[used[cnt]][i]<<",";
		cout<<endl; 
		trainnum++;
		if(trainnum==max_trainnum)
		{
			for(int i=0;i<MAXN;i++){
    			w[used[cnt]][i]=(int)(100000.0*w[used[cnt]][i])/100000.0;
			}
			trainnum=0;
			break;
		}
	}
}*/
void fitting() {
    // 假设目标是最小化均方误差
    for (int i = 0; i < MAXN; ++i) {
        float gradient = 0;
        // 计算偏导数(gradient)
        for (int j = 0; j <= ALL; j++) {
            gradient += (h[j] - x[j][i]) * x[j][i];
        }
        // 考虑到样本数量对梯度的缩放
        gradient /= (ALL + 1);
        
        // 更新权重
        for (int j = 0; j <= ALL; j++) {
            // 注意，这里使用了负梯度方向更新权重
            w[j][i] -= alpha * gradient;
        }
    }
    
    // 更新学习率（如果需要）
    alpha *= (1.0 - alphaDecayRate);
    if (alpha < minAlpha) {
        alpha = minAlpha;
    }
}
int main()
{
	//0 1 2
    used[0] = 0;
    used[1] = 1;
    cnt = 1;
    feature();
    toh();
    softmax();
    float maxx=0;
    for(int i=0; i<ALL; i++)
    {
        if(pro[i] > maxx)
        {
            maxx = pro[i];
            pos = i;
        }
    }
    for(int i=0; i<ALL; i++)
    {
        cout<<pro[i]<<endl;
    }
    reall = 2;
	fitting();
    while(1){
	    feature();
	    toh();
	    softmax();
	    float maxx=0;
	    for(int i=0; i<ALL; i++)
	    {
	        if(pro[i] > maxx)
	        {
	            maxx = pro[i];
	            pos = i;
	        }
	    }
	    for(int i=0; i<ALL; i++)
	    {
	        cout<<pro[i]<<endl;
	    }
	    if(pos!=reall){
	    	loss=pro[pos]-pro[reall];
	    	cout<<"loss:"<<loss<<endl;
	    	fitting();
		}
		else if(pos==reall) break;
	}
    cout<<"Done."<<endl;
    for(int i=0;i<MAXN;i++){
    	cout<<w[used[cnt]][i]<<","; 
	}
    return 0;
}
