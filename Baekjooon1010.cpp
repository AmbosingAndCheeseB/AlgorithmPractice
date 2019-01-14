#include <iostream>

using namespace std;

int main(){
	
	int test, i;
	
	cin >> test;
	
	int *n = new int[test];
	int *m = new int[test];
	
	for(i=0; i<test; i++)
		cin >> n[i] >> m[i];
		
	//code
	
		
	delete[] n;
	delete[] m;
	
	return 0;
}
