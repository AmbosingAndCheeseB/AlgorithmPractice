#include <iostream>

using namespace std;

int getInt();

int main(){
	
	int step, width, space, root;
	cout << "단 수를 입력하세요.(1에서 50 사이 정수):";
	
	while(true){
		step = getInt();
		if(step>=1 && step<=50) break;
		cout << "1에서 50사이로 다시 입력하세요.:"; 
	}
	
	width = 2*step + 3;
	
	for(int i=1; i<=step; i++){
	
		space = width/2;
		
		for(int j=1; j<=i+2; j++){
		
			for(int k=1; k<=space; k++) cout << " ";
			for(int l=1; l<=2*j-1; l++) cout << "*"; 
			
			space=space-1;	
			cout << endl;
		}
	}
	
	
	root = width/2;
	
	for(int j=1; j<=root; j++){
		for(int k=1; k<=root; k++) cout << " ";
		cout << "*";
		
		cout << endl;
	}
	
	
	return 0;
}

int getInt(){
	
	int num;
	float f;
	
	cin >> f;
	if(cin.fail()){
		cout << "문자 말고 정수를 입력하세요.:";
		cin.clear();
		cin.ignore();
		return getInt();
	}
	else{
		num = static_cast<int>(f);
		
		if(static_cast<float>(num) != f){
			cout << "실수 말고 정수를 입력하세요.:";
			return getInt(); 
		}
		else return num;
	}
}
