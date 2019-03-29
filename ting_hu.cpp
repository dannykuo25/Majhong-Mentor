# include <iostream>
# include <algorithm>
# include <vector>
# include <fstream>
# include <typeinfo>
using namespace std;
bool canhu ( vector<int>& , vector<int>& );
	
int main(int argc, char** argv){
	#define SIZE 100
	char line[SIZE];
	// input into in_vec
    vector<int> in_vec; 
	fstream file;
    file.open("input.txt",ios::in);
    while(file.getline(line,sizeof(line),' ')){
		int iline = atoi(line);
		in_vec.push_back(iline);
    }
	// ting case
	if( in_vec.size() == 13){
		vector<int> ting;
		for ( int p = 1; p <= 9; p++ ){
			// add one tile per time
			in_vec.push_back( p );
			// sort
			sort( in_vec.begin(), in_vec.end() );
			// count numbers of each type of tiles
			vector<int> count ( 10 ); 
			for ( int i = 0; i < in_vec.size(); i++ ){
				for ( int j = 1; j <= 9; j++ ){
					if ( in_vec[i] == j ) {
						count[j]++;
						break;
					}
				}
			}
			
			if ( canhu( in_vec, count ) == 1 ){
				ting.push_back( p );
			}
			
			// delete the tile added before
			for ( int i = 0; i < in_vec.size(); i++ ){
				if ( in_vec[i] == p ){
					in_vec.erase( in_vec.begin() + i );
					break;
				}
			}
			
		}
		ofstream fout;
		fout.open("output.txt");
		for ( int i = 0; i < ting.size(); i++ ){
			fout << ting[i] << " ";
		}
		fout.close();
	}
	// hu case
	else if ( in_vec.size() == 14 ){
		if ( canhu( in_vec, count))
	}
	else{
		ofstream fout;
		fout.open("output.txt");
		fout << "Wrong input!\n";
		fout.close();		
	}
	
	return 0;
}

bool canhu ( vector<int>& ptr, vector<int>& cnt_ptr ){
	bool win = 0;
	for ( int j = 1; j <= 9; j++ ){
		if ( win == 1 ) break;
		// find a pair of tiles
		if ( cnt_ptr[j] >= 2 ){	
			//copy vector
			vector<int> a_new( ptr.size() );
			for ( int i = 0; i < ptr.size(); i++ )
				a_new[i] = ptr[i];
				
			//define which index to remove from vector
			int remove1 = 0;
			int remove2 = 0;
			for ( int i = 0; i < a_new.size(); i++ ){
				if ( a_new[i] == j ) {
					remove1 = i;
					remove2 = i + 1;
					break;
				}
			}
            // remove a pair of tiles
			a_new.erase( a_new.begin() + remove2 );
			a_new.erase( a_new.begin() + remove1 );
			
		
			while ( a_new.size() != 0 ){	
				
				// if it is a flush, such as x,x,x or y,y,y	
				if ( ( a_new[0] == a_new[1]) && ( a_new[1] == a_new[2] ) ){
					a_new.erase( a_new.begin() );
					a_new.erase( a_new.begin() );
					a_new.erase( a_new.begin() );
                    // if no tiles left, win
					if ( a_new.size() == 0 ){
						win = 1;
					}
				}
				
				// if it's a straight, such as x, x+1, x+2 
				// else case, try to find x, x+1, x+2
				else{  
					int rv1 = 0, rv2 = 0; // index of x+1, x+2
					int flag = 0; // to confirm there exists x, x+1, x+2
					for ( int i = 1; i < a_new.size(); i++ ){
						flag = 0; 					
						
						//find that x+1 exist
						if ( a_new[i] == a_new[0] + 1 ){
							rv1 = i;
							for ( int k = rv1 + 1; k < a_new.size(); k++ ){
								//find that x+2 exist
								if ( a_new[k] == a_new[0] + 2 ){ 
									flag = 1;
									rv2 = k;
									//delete x, x+1, x+2
									a_new.erase( a_new.begin() + rv2 );
									a_new.erase( a_new.begin() + rv1 );
									a_new.erase( a_new.begin() );
									//if there is no remaining tile, hu!
									if ( a_new.size() == 0 ){
										win = 1;
									}
									break;								
								}								
							}
							break;
						}						
					}
					// no x, x+1, x+2 exists
					if ( flag == 0 )	break;											
				}
			}
		// renew copy vector
		a_new.clear();
		}
	}
	// show result
	if ( win == 1 )	return 1;
	else return 0;

}
