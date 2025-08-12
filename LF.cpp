#include<bits/stdc++.h>
#include<windows.h>
#include<cmath>
#define MAXN 5
using namespace std;
double w[3][MAXN]={
{0.1254,0.7635,-0.2973,-0.9725,0.4523},
{0.7467,0.8460,-0.4659,-0.0172,0.1566},
{0.5851,0.12459,-0.7596,-0.9194,0.2789}
};
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
double loss,ploss;
double delta;
double alpha=0.01;//learning rate
int fv[MAXN][MAXN],fcnt=-1; 
double present[MAXN][MAXN];
double maxv[MAXN][MAXN];//best upgrade
int cols=MAXN;
bool endlf=false;
double transmoid(double x)
{
    double absX=abs(x);
    double powerOfTwo=pow(2.0, -absX);
    double subtractOne = powerOfTwo - 1;
    double sign = (x >= 0) ? -1 : 1;
    double y = sign*(x/absX)*subtractOne;
    return y;
}
void feature()
{
	for(int i=0;i<5;i++)
	{
		for(int j=0;j<cnt;j++)
		{
			f[i]=w[used[j]][i]*x[used[j]][i];
		}
	}
}
void toh()
{
	for(int i=0;i<5;i++)
	{
		h[i]=transmoid(f[i]);
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
			dot_product+=h[i]*x[i][j];
		}
		pro[i]=exp(dot_product);
		sum_exp+=pro[i];
	}
	for(int i=0;i<3;i++)
	{
		pro[i]/=sum_exp;
	}
}
double x_add_w(int t)
{
	double temp=0;
	for(int i=0;i<5;i++){
		temp+=x[t][i]*w[t][i];
		cout<<temp<<endl; 
	} 
	return temp;
}
void fitting(){
    //fitting
    for(int k=0;k<=cnt;k++){
    	for(int a=0;a<5;a++){//vectors upgrading
			w[used[k]][a]+=delta*fv[k][a];
		}
	}
	feature();
	toh();
	softmax();
	loss=pro[pos]-pro[reall];
	if(loss<ploss&&loss>=0){
		for(int k=0;k<=cnt;k++){//max value saving
			for(int b=0;b<5;b++){
				maxv[k][b]=w[k][b];
			}
		}
		ploss=loss;
	}
	else if(loss<ploss&&loss<0){
		endlf=true;
		return;
	}
	cout<<"Loss:"<<loss<<endl;
	Sleep(50);
	for(int k=0;k<=cnt;k++){
		for(int a=0;a<5;a++){
			w[k][a]=present[k][a];
		}
	}
}
void FittingControl(){
	int rows = pow(3, cols);
	for (int i = 0; i < rows; i++) {
        int temp = i;
        for(int k=0;k<=cnt;k++){
        	for (int j = 0; j < cols; j++) {
        	    int value = temp % 3 - 1;
        	    fv[k][++fcnt]=value;
        	    temp /= 3;
        	}
			fcnt=0;
		}
		fitting();
		if(endlf==true){
			return;
		}
    }
}
void lf()
{
	loss=pro[pos]-pro[reall];
	double z=x_add_w(reall); 
	cout<<"Z:"<<z<<endl;
	delta=alpha*(loss*100/z);
	cout<<"Loss:"<<loss<<" Delta:"<<delta<<endl;
	ploss=loss;
	FittingControl();
}
int main()
{
	used[0]=1;
	used[1]=2;
	cnt=1;
	feature();
	toh();
	softmax();
	double maxx=0;
	for(int i=0;i<3;i++)
	{
		//cout<<pro[i]<<endl;
		if(pro[i]>maxx){
			maxx=pro[i];
			pos=i;
		}
	}
	reall=2;
	while(1)
	{
		lf();
		for(int k=0;k<=cnt;k++){
			for(int a=0;a<5;a++){
				w[k][a]=maxv[k][a];
			}
		}
		if(endlf==true){
			endlf=false;
			break;
		}
	}
	return 0;
}
