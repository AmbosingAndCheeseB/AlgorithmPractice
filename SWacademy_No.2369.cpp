#include <iostream>

using namespace std;

int main(){
	
	int t, n, l, r, seat =0;
	
	cin >> t;
	for(int i=1; i<=t; i++){
		cin >> n;
		for(int j=0; j<n; j++){
			cin >> l >> r;
			seat += r-l+1;
		}
		cout << '#' << i << " " << seat << endl;
		seat = 0;
	}
	
	return 0;
}
