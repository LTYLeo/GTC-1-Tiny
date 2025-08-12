#include<bits/stdc++.h>
using namespace std;
float delta;
int main(){
	cin>>delta;
	if(delta>0){
		if(delta<0.00001){
			double t=1;int numdiv;
			for(int i=1;i<=9;i++){
				t=t/10;
				cout<<t<<endl;
				if(delta>t){
					numdiv=i-5;
					break;
				}
			}
			cout<<numdiv<<endl;
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
					numdiv=i;
					break;
				}
			}
			for(int i=1;i<=numdiv;i++){
				delta=delta*10;
			}
		} 
		delta=-delta;
	}
	cout<<delta;
	return 0;
}
