#include<bits/stdc++.h>
#include<windows.h>
#include<cmath>
#define MAXN 10
#define ALL 3
using namespace std;
float w[ALL][MAXN]={
//{0.1254,0.7635,-0.2973,-0.9725,0.4523,0.0024},
{-0.24307,-0.83282,-0.09142,-0.75026,-0.7553,-0.09447,-0.9322,-0.01826,-0.30716,-0.05628},
{0.56988,-0.18237,0.75242,0.875,-0.43793,0.3336,-0.5263,0.2635,-0.4234,0.1333},
{-0.9322,-0.01826,-0.30716,-0.05628,0.22472,0.2712,0.56988,-0.18237,0.75242,0.875}
};
float x[ALL][MAXN]={
    {0.2345,0.6534,0.8542,-0.1233,0.7635,0.8384,-0.5263,0.2635,-0.4234,0.1333},
    {-0.5263,0.2635,-0.4234,0.1333,0.4934,-0.2398,0.2134,-0.4568,0.3248,0.2189},
    {0.2134,-0.4568,0.3248,0.2189,-0.9742,-0.3421,0.2345,0.6534,0.8542,-0.1233}
};
queue<int>q; 
string wl[ALL]={"1","2","3"};
int trainnum=0,max_trainnum=50; 
float f[MAXN];
float h[MAXN];
int used[100];
double pro[ALL];
int cnt=-1;
int reall;
int pos;
float loss,minloss=1e9,preloss;
float delta;
float alpha = 0.1;
float minAlpha = 0.01;
float alphaDecayRate = 0.000001;
int fv[MAXN],fcnt=-1; 
float present[MAXN];
float maxv[MAXN];
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
}

float beta1 = 0.9;
float beta2 = 0.999;
float epsilon = 1e-8;
int t = 0; // 用于记录迭代次数

// Adam 优化器的矩估计
float m[ALL][MAXN] = {0};
float v[ALL][MAXN] = {0};

void fitting() {
    if (fv[0] == 0 && fv[1] == 0 && fv[2] == 0 && fv[3] == 0 && fv[4] == 0) {
        return;
    }

    t++; // 增加迭代次数

    feature();
    toh();
    softmax();

    float gradient[MAXN]; // 梯度
    memset(gradient, 0, sizeof(gradient));

    // 计算梯度
    for (int a = 0; a < MAXN; a++) {
        float grad = 0;
        for (int i = 0; i <= ALL; i++) {
            grad += (pro[i] - (i == reall ? 1 : 0)) * x[i][a];
        }
        gradient[a] = grad;
    }

    // 更新权重
    for (int a = 0; a < MAXN; a++) {
        // 更新一阶和二阶矩估计
        m[used[cnt]][a] = beta1 * m[used[cnt]][a] + (1 - beta1) * gradient[a];
        v[used[cnt]][a] = beta2 * v[used[cnt]][a] + (1 - beta2) * gradient[a] * gradient[a];

        // 偏差校正
        float m_hat = m[used[cnt]][a] / (1 - pow(beta1, t));
        float v_hat = v[used[cnt]][a] / (1 - pow(beta2, t));

        // 更新权重
        w[used[cnt]][a] -= alpha * m_hat / (sqrt(v_hat) + epsilon);
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
    }
}

void lf()
{
	int count=0;
	loss = pro[pos] - pro[reall];
	float z = 0.0;
	for (int i = 0; i < MAXN; i++)
	{
	    z += x[pos][i] * w[pos][i];
	}
	delta = alpha * (loss / z) ;
	cout << "Loss:" << loss << " Delta:" << delta << endl;
	while(1){
		for(int i=0;i<MAXN;i++){
			present[i]=w[used[cnt]][i]; 
		}
		FittingControl();
		alpha -= alphaDecayRate;
        if (alpha < minAlpha) {
            alpha = minAlpha;
        }
        z=0.0;
        for (int i = 0; i < MAXN; i++)
		{
		    z += x[pos][i] * w[pos][i];
		}
        delta = alpha * (minloss / z) * 10;
		for(int a=0;a<MAXN;a++)
		{
			w[used[cnt]][a] = maxv[a];
		}
		if(minloss==preloss){
			count++;
			preloss=minloss;
		}
		if(delta>0){
			if(delta<0.00001){
				double t=1;int numdiv;
				for(int i=1;i<=9;i++){
					t=t/10;
					if(delta>t){
						numdiv=i-5;
						break;
					}
				}
				for(int i=1;i<=numdiv;i++){
					delta=delta*10;
				}
			} 
		}
		else if(delta<0){
			delta=abs(delta);
			if(delta<0.00001){
				double t=1;int numdiv;
				for(int i=1;i<=9;i++){
					t=t/10;
					if(delta>t){
						numdiv=i-5;
						break;
					}
				}
				for(int i=1;i<=numdiv;i++){
					delta=delta*10;
				}
			} 
			delta=-delta;
		}
		//cout<<"Trainnum:"<<trainnum << " Loss:" << minloss << " Delta:" << delta << endl;
		//Sleep(25);
		/*for(int i=0;i<MAXN;i++){
    		cout<<maxv[i]<<" "; 
		}*/
		preloss=minloss;
		//cout<<endl;
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
}

int main()
{
	string input;
	while(1){
		getline(cin,input);
		if(input=="<END>") break;
		for(int i=0;i<input.size();i++){
			for(int j=0;j<ALL;j++){
				string str;
				str=input[i];
				if(str==wl[j]){
					q.push(j);
					break;
				}
			}
		}
		q.push(-1);
		while(!q.empty()){
			used[++cnt]=q.front();
			q.pop();
			if(q.front()==-1){
				break;
			}
			reall=q.front();
			//
			int processnum=0;
		    while(1){
		    	processnum++;
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
			    	cout<<"Process:"<<processnum<<endl;
			    	lf();
				}
				else if(pos==reall) break;
			}
			cout<<endl<<"W:"<<endl;
			for(int i=0;i<ALL;i++){
				cout<<i<<":{";
				for(int j=0;j<MAXN;j++){
					cout<<w[i][j]<<",";
				}
				cout<<"}"<<endl;
			}
		}
		cnt=-1;
	}
    return 0;
}
