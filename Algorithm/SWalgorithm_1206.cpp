#include <iostream>

using namespace std;

int main(){
	
	int building[10][1001]; 
	int i,j;
	
	/*테스트케이스 입력*/
	for(i=0; i<10; i++){
		cin >> building[i][0]; //건물 수 
		for(j=1; j<=building[i][0]; j++){
			cin >> building[i][j]; //각 건물 높이 
		}
	}
	
	
	
	return 0;
}
