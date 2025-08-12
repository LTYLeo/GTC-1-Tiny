#include<bits/stdc++.h>
#include<cmath>
#include<Windows.h>
#define MAXN 6
#define ALL 5
#define maxtext 10
using namespace std;
float w[ALL][MAXN]={
{-9.24307,-4.83282,-7.09142,-4.75026,-2.7553,-4.09447},
{5.56988,-7.18237,8.75242,10.875,-8.43794,10.3336},
{-11.9322,-1.01826,-3.30716,-8.05628,9.22472,11.2712},
{-10.5263,-6.2635,-9.4234,-8.1333,3.4934,-5.2398},
{2.2134,-3.4568,7.3248,12.2189,-7.9742,-3.3421}
};
float x[ALL][MAXN]={
    {0.2345,0.6534,0.8542,-0.1233,0.7635,0.8384},
    {-0.5263,0.2635,-0.4234,0.1333,0.4934,-0.2398},
    {0.2134,-0.4568,0.3248,0.2189,-0.9742,-0.3421},
    {0.56988,-0.18237,-0.75242,-0.875,-0.43794,0.3336},
	{-0.9322,-0.01826,-0.30716,-0.05628,0.22472,0.2712}
};
queue<int>q; 
string wl[ALL]={"I","am","a","human","."};
float f[MAXN];
float h[MAXN];
int used[100];
double pro[ALL];
int cnt=-1;
int pos;
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
/*void softmax()
{
    double sum_exp=0;
    for(int i=0;i<ALL;i++)
    {
        double dot_product=0;
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
}*/
void softmax()
{
    double sum_exp=0;
    double max_val = -INFINITY;
    for(int i=0;i<ALL;i++)
    {
        double dot_product=0;
        for(int j=0;j<MAXN;j++)
        {
            dot_product += h[j] * x[i][j];
        }
        pro[i] = exp(dot_product);
        if (dot_product > max_val) {
            max_val = dot_product;
        }
        sum_exp += pro[i];
    }
    for(int i=0;i<ALL;i++)
    {
        pro[i] = exp(pro[i] - max_val) / sum_exp;
    }
}

int main(){
	int pos;
	string str;
	while(1){
		getline(cin,str);
		string s;
		if(str=="<END>") break;
		for(int i=0;i<str.size();i++){
			if(str[i]==' '){
				for(int j=0;j<ALL;j++){
					if(s==wl[j]){
						q.push(j);
						break;
					}
				}
				s="";
			}
			else s+=str[i];
		}
		while(!q.empty()){
			used[++cnt]=q.front();
			q.pop();
		}
		cout<<str;
		for(int i=1;i<=maxtext;i++){
			feature();
			toh();
			softmax();
			double maxx=0.0;
			for(int i=0;i<ALL;i++){
				if(pro[i] > maxx)
				{
					maxx = pro[i];
					pos = i;
				}
			}
			cout<<endl;
			for(int i=0;i<ALL;i++){
				cout<<pro[i]<<endl;
			}
			cout<<" "<<wl[pos];
			used[++cnt]=pos;
			Sleep(500);
		}
		cnt=-1;
		cout<<endl;
	}
	return 0; 
}
