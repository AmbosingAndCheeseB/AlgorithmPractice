#include <iostream>

using namespace std;

int main(){
	
	int test, i, j, fac, per;
	
	cin >> test;
	
	int *n = new int[test];
	int *m = new int[test];
	int *ans = new int[test];
	
	for(i=0; i<test; i++)
		cin >> n[i] >> m[i];
		
	for(i=0; i<test; i++){
		
		fac = 1;
		per = 1;
		
		for(j=n[i]; j>0; j--)
			fac *= j;
			
		for(j=m[i]; j>m[i]-n[i]; j--)
			per *= j;
			
		ans[i] = per/fac;
	
	}
	
	for(i=0; i<test; i++)
		cout << ans[i] << endl;
	
		
	delete[] n;
	delete[] m;
	delete[] ans;
	
	return 0;
}
