#include<bits/stdc++.h>
#include<windows.h>
#include<cmath>
#define MAXN 5
using namespace std;
double w[3][MAXN]={
{0.1254,0.7635,-0.2973,-0.9725,0.4523},
{0.7467,0.8460,-0.4659,-0.0172,0.1566},
{0.5851,0.1245,-0.7596,-0.9194,0.2789}
};//7.14121,-7.94534,4.92625,8.77414,-8.63474
double x[3][MAXN]={
    {0.2345,0.6534,0.8542,-0.1233,0.7635},
    {-0.5263,0.2635,-0.4234,0.1333,0.4934},
    {0.2134,-0.4568,0.3248,0.2189,-0.9742}
};
string wl[3]={"1","2","3"};

double f[MAXN];
double h[MAXN];
int used[MAXN];
double pro[3];
int cnt=-1;
int reall;
int pos;
double loss,minloss=1e9,preloss;
double delta;
double alpha = 0.1; // 初始学习率
double minAlpha = 0.01; // 最小学习率
double alphaDecayRate = 0.000001; // 学习率衰减率
int fv[MAXN],fcnt=-1; 
double present[MAXN];
double maxv[MAXN];//best upgrade
int cols=MAXN;
bool endlf=false;
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
void feature()
{
    for(int i=0;i<5;i++)
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
    for(int i=0;i<5;i++)
    {
        h[i] = sigmoid(f[i]);
    }
}

void softmax()
{
    double sum_exp=0;
    for(int i=0;i<3;i++)
    {
        double dot_product=0;
        for(int j=0;j<5;j++)
        {
            dot_product += h[j] * x[i][j];
        }
        pro[i] = exp(dot_product);
        sum_exp += pro[i];
    }
    for(int i=0;i<3;i++)
    {
        pro[i] /= sum_exp;
    }
}

void fitting()
{
	if(fv[0]==0&&fv[1]==0&&fv[2]==0&&fv[3]==0&&fv[4]==0){
		return;
	}
    // Fitting
	for(int a=0;a<5;a++)
	{
		if(fv[a]==1){
			w[used[cnt]][a] += delta;
		}
		else if(fv[a]==-1){
			w[used[cnt]][a] -= delta;
		}
	}
    feature();
    toh();
    softmax();
    double maxx=0;
    for(int i=0; i<3; i++)
    {
        if(pro[i] > maxx)
        {
            maxx = pro[i];
            pos = i;
        }
    }
    loss = pro[pos] - pro[reall];
    if(loss < minloss)
    { 
		for(int b=0;b<5;b++)
		{
			maxv[b]=w[used[cnt]][b];
		}
        minloss = loss;
    }
    if(loss < 0)
    {
        endlf=true;
        return;
    }
	for(int a=0;a<5;a++)
	{
		w[used[cnt]][a] = present[a];
	}
}

void FittingControl()
{
    int rows = pow(3, MAXN);
    for (int i = 0; i < rows; i++)
    {
        int temp = i;
        for (int j = 0; j < MAXN; j++)
        {
            int value = temp % 3 - 1;
            fv[j]=value;
            temp /= 3;
        }
        fitting();
        if(endlf==true)
        {
            return;
        }
    }
}

void lf()
{
	int count=0;
	loss = pro[pos] - pro[reall];
	if (loss <= 0)
	{
	    endlf = true;
	    return;
	}
	double z = 0.0;
	for (int i = 0; i < 5; i++)
	{
	    z += x[pos][i] * w[pos][i];
	}
	delta = alpha * (loss / z) *10;
	cout << "Loss:" << loss << " Delta:" << delta << endl;
	while(!endlf){
		for(int i=0;i<5;i++){
			present[i]=w[used[cnt]][i]; 
		}
		FittingControl();
		alpha -= alphaDecayRate;
        if (alpha < minAlpha) {
            alpha = minAlpha;
        }
        z=0.0;
        for (int i = 0; i < 5; i++)
		{
		    z += x[pos][i] * w[pos][i];
		}
        delta = alpha * (minloss / z) * 10;
        if(loss<0.00001){
			for(int i=0;i<5;i++){
    			maxv[i]=(int)(100000.0*maxv[i])/100000.0;
			}	
		}
		for(int a=0;a<5;a++)
		{
			w[used[cnt]][a] = maxv[a];
		}
		if(minloss==preloss){
			count++;
			preloss=minloss;
		}
		cout << "Loss:" << minloss << " Delta:" << delta << endl;
		Sleep(25);
		for(int i=0;i<5;i++){
    		cout<<maxv[i]<<" "; 
		}
		preloss=minloss;
		/*if(count>=3){
			for(int i=0;i<5;i++){
				w[used[cnt]][i]-=0.5;
			}
			count=0;
		}*/ 
		cout<<endl;
	}
}

int main()
{
    used[0] = 0;
    used[1] = 1;
    cnt = 1;
    feature();
    toh();
    softmax();
    double maxx=0;
    for(int i=0; i<3; i++)
    {
        if(pro[i] > maxx)
        {
            maxx = pro[i];
            pos = i;
        }
    }
    reall = 2;
    lf();
    cout<<"Done."<<endl;
    for(int i=0;i<5;i++){
    	cout<<w[used[cnt]][i]<<" "; 
	}
    return 0;
}

