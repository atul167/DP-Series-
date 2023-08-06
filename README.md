# Algo-Series
Sheet 
1. Partition equal sum : https://leetcode.com/problems/partition-equal-subset-sum/description/
```
// This is my first DP code on Github
class Solution {
public:
int n;
bool check=false;
    bool canPartition(vector<int>& nums) {
        n=nums.size();
        int sum1=accumulate(nums.begin(),nums.end(),0);
        if(sum1%2==1)
            return false;
        int sum=sum1/2;
        bool dp[n+1][sum+1];
        for(int i=0; i<=n;i++)
            dp[i][0]=1;
        for(int i=0; i<=sum;i++)
            dp[0][i]=0;
        for(int i=1; i<=n; i++)
        {
            for(int j=1; j<=sum ;j++)
            {
                dp[i][j]=dp[i-1][j];
                if(j>=nums[i-1])
                {
                  int x=dp[i-1][j];
                  int y=dp[i-1][j-nums[i-1]];
                  dp[i][j]=x|y;
              }
          }
      }
      return dp[n][sum];
    }
};
----------------------------------------------------------
#include <bits/stdc++.h> 
int dp[1001][10001];
bool solve(int pos,int n,int k,vector<int>&arr,int sum){
  if (pos == n) {
    if (sum == k)
      return true;
    return false;
  }
  if (dp[pos][sum] != -1)
    return dp[pos][sum];
  return dp[pos][sum]=solve(pos + 1, n, k, arr, sum + arr[pos])|solve(pos+1,n,k,arr,sum);
}
bool subsetSumToK(int n, int k, vector<int> &arr) {
  memset(dp,-1,sizeof(dp));
    return solve(0,n,k,arr,0);
}
```
2. Longest Increasing subsequence (finding the length and printing the subsequence):https://leetcode.com/problems/longest-increasing-subsequence/
```
class Solution {
public:
#define ll long long 
 int lengthOfLIS(vector<int>& arr) {
   int n=arr.size();
   int dp[n];
   int hash[n];// for hashing the index
   for(int i=0;i<n;i++)
   hash[i]=i;
   int lst=-1;
   for(int i=0;i<n;i++)
   {
       dp[i]=1;
       for(int j=0;j<i;j++)
       {
           if(arr[i]>arr[j]&&dp[i]<=(dp[j]+1))
           {
               dp[i]=dp[j]+1;
               hash[i]=j;
           }
       }
   }
   int maxi=0;
   for(int i=0;i<n;i++)
   {
     if (maxi <= dp[i]) {
       maxi = dp[i];
       lst = i;
     }
   }
   vector <int> v;
   while (hash[lst] != lst)
    {
        v.push_back(arr[lst]);
        lst=hash[lst];
    }
    for(int i=0;i<n;i++)
    cout<<dp[i]<<" ";
    cout<<endl;
    return maxi;
    }
};
```
NlogN method:

![image](https://github.com/atul167/DP-Series-/assets/76389640/8c0a0af9-c628-4332-9bfe-576cde90e85b)
```
for (int i = 0; i < n;i++){
		cin >> a[i];}
	de(a);
	dp.push_back(a[0]);
	for(int i=1;i<n;i++){
		int pos = lower_bound(dp.begin(), dp.end(), a[i]) - dp.begin();
		if(pos==dp.size()){
			//we can have a longer lis
		dp.pb(a[i]);
		}
		else
		dp[pos] = a[i];//we can have smaller element at the last
		de(dp);
	}
	de(dp);
	cout << dp.size() << endl;
    
```
3.Buy and sell stock (3) :https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
```

class Solution {
public:
int n;
    int maxProfit(vector<int>& prices) {
        n=prices.size();
        vector<vector<vector<int>>>dp(n+1,vector<vector<int>>(2,vector<int>(3,0)));
        int buy=1,cap=2;
        for(int i=n-1;i>=0;i--)
        {
            for(int buy=0;buy<=1;buy++)
            {
                for(int cap=1;cap<=2;cap++)
                {
                    if(buy)
                    dp[i][buy][cap]=max(-prices[i]+dp[i+1][buy^1][cap],dp[i+1][buy][cap]);// buy or skip
                    else
                    dp[i][buy][cap]=max(prices[i]+dp[i+1][buy^1][cap-1],dp[i+1][buy][cap]);// sell or skip ,cap value changes if sold as one transaction has been done
                }

            }
        }
        return dp[0][1][2];
    }
};
```
4.Trapping rain water problem :https://leetcode.com/problems/trapping-rain-water/
```
class Solution {
public:
    int trap(vector<int>& height) {
        int n=height.size();
        int left[n];
        int right[n];
        left[0]=height[0];
        right[n-1]=height[n-1];
        for(int i=1;i<=n-1;i++){
             left[i]=max(left[i-1],height[i]);
        }
        for(int i=n-2;i>=0;i--){
            right[i]=max(right[i+1],height[i]);
        }
       
        int ans=0;
        for(int i=1;i<=n-2;i++)
        {
            int mn=min(right[i+1],left[i-1]);
            int rem=mn-height[i];
            if(rem>=0)
            ans+=rem;
        }
        return ans;

    }
};
```
5. Distinct subsequence :https://leetcode.com/problems/distinct-subsequences-ii/description/
```
class Solution {
public:
#define ll long long 
const ll md=1e9+7;
    int distinctSubseqII(string s) {
        map<char,int>last;
        int n=s.size();
        vector<ll>dp(n+1,0);
        dp[0]=1;
        for(int i=1;i<=n;i++){
            dp[i]=(1ll*2*dp[i-1])%md;
            if(last.count(s[i-1]))
            {
                int x=last[s[i-1]];
                dp[i]-=x;
            }
            dp[i]=(dp[i]+md)%md;           
            last[s[i-1]]=dp[i-1];
        }
        ll x=dp[n]-1;
        x=(x+md)%md;
        return x;
    }
};
```
6.Poisonous Full Course :https://atcoder.jp/contests/abc306/tasks/abc306_d
```
#include<bits/stdc++.h>
using namespace std;
#define fastio() ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL)
#define mod 1000000007
#define pb push_back
#define mp make_pair
#define ff first
#define ss second
#define set_bits __builtin_popcountll
#define all(x) (x).begin(), (x).end()
typedef long long ll;
typedef long double lld;
const int md = 998244353;
const int mxn = 50001;
int main() {
	ll n;
	cin >> n;
	vector<pair<ll,ll>>vec(n);
	for(int i=0;i<n;i++){
		cin>>vec[i].ff>>vec[i].ss;
	}
	ll dp[n+1][2];
	for(int i=0;i<=n;i++){
		dp[i][0]=-4e18;
		dp[i][1]=-4e18;
	}
	dp[0][0]=0;
	ll health=0;
	for(int i=0;i<n;i++){
		ll type=vec[i].ff;
		ll f=vec[i].ss;
		//eating the choice
		if(type==0){
			dp[i+1][0]=max(max(dp[i][0],dp[i][1])+f,dp[i][0]);
		}
		else{
			dp[i+1][1]=max(dp[i][0]+f,dp[i][1]);
		}
		//skipping the choice 
		dp[i+1][0]=max(dp[i][0],dp[i+1][0]);
		dp[i+1][1]=max(dp[i][1],dp[i+1][1]);
		
	}
	cout<<max(dp[n][0],dp[n][1])<<endl;
	return 0;
}
```
7.https://cses.fi/problemset/task/1679 (Based on Topological Sort):
```
#include<bits/stdc++.h>
using namespace std;
#define fastio() ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL)
#define mod 1000000007
#define pb push_back
#define ff first
#define ss second
#define all(x) (x).begin(), (x).end()
typedef long long ll;
typedef long double lld;
const int mxn = 2e5 + 100; 
stack<ll>s;
vector<bool>vis;
vector<vector<ll>>adj(mxn);
void dfs(int v,vector<vector<ll>>&adj){
	vis[v]=1;
	for(auto u:adj[v]){
		if(!vis[u]){
			dfs(u,adj);
		}
	}
	s.push(v);
}
void pr(){
	cout<<"IMPOSSIBLE"<<endl;
}
int main() {
ll tt;
tt=1;
while (tt--)
{
	ll n,m;
	cin>>n>>m;
	vis=vector<bool>(n);
	for(int i=0;i<m;i++){
		ll u,v;
		cin>>u>>v;
		--u;--v;
		adj[u].pb(v);
	}
	vector<ll>ans;
	for(int i=0;i<n;i++){
		if(!vis[i])
		dfs(i,adj);
	}
	while(!s.empty()){
		auto x=s.top();
		ans.pb(x);
		s.pop();
	}
	vector<ll>index(n);
	for(int i=0;i<n;i++){
		index[ans[i]]=i;
	}
	for(int i=0;i<n;i++){
		for(auto u:adj[i]){
			if(index[i]>index[u])
			{
				pr();
				return 0;
			}
		}
	}
	for(auto u:ans)
	cout<<u+1<<" ";
	cout<<endl;
}
return 0;
}
```
8: Digit Sum between L to R(https://www.spoj.com/problems/PR003004/)
```
#include<bits/stdc++.h>
using namespace std;
#define fastio() ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL)
#define mod 1000000007
#define pb push_back
#define mp make_pair
#define ff first
#define ss second
#define set_bits __builtin_popcountll
#define all(x) (x).begin(), (x).end()
typedef long long ll;
typedef long double lld;
const int mxn = 1e5 + 100;
ll dp[20][200][2];
string str;
ll solve(ll pos,ll sum,ll tight){
	if(pos==str.size())
		return sum;
	if(dp[pos][sum][tight]!=-1)
		return dp[pos][sum][tight];
	int ub = 0;
	if (tight)
		ub = str[pos] - '0';
	else
		ub = 9;
	ll ans = 0;
	for (int i = 0; i <= ub;i++){
		ans += solve(pos + 1, sum + i, (tight & (i == ub)));
	}
	return dp[pos][sum][tight] = ans;
}
ll go(ll x){
	str = to_string(x);
	memset(dp, -1, sizeof(dp));
	return solve(0, 0, 1);
}
int main() {
	fastio();
	ll tt;
	cin >> tt;
	while (tt--)
	{
		ll a, b;
		cin >> a >> b;
		memset(dp, -1, sizeof(dp));
			cout << go(b) - go(a - 1) << endl;
}
return 0;
}
```
9.Cherry Pickup :https://leetcode.com/problems/cherry-pickup-ii/
```
class Solution {
public:
#define ll long long 
int dp[73][73][73];
int n,m;
int  solve(int row,int col1,int col2,vector<vector<int>>&grid)
{
    if(row>=n||col1<0||col2<0||col1>=m||col2>=m)
    return 0;
    if(dp[row][col1][col2]!=-1)
    return dp[row][col1][col2];
   int res=0;
   res+=grid[row][col1];
   if(col1!=col2)
   res+=grid[row][col2];
   int mx=0;
   
       for(int i=col1-1;i<=col1+1;i++)
       {
           for(int j=col2-1;j<=col2+1;j++)
           mx=max(mx,solve(row+1,i,j,grid));
       }
 
   res+=mx;
   return dp[row][col1][col2]=res;
}
int cherryPickup(vector<vector<int>>& grid) {
        n=grid.size();
        m=grid[0].size();
        memset(dp,-1,sizeof(dp));
       return  solve(0,0,m-1,grid);
    }
};

```
10. Grid Paths :https://leetcode.com/problems/unique-paths/
```
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>>dp(m,vector<int>(n,0));
        for(int i=0;i<n;i++){
            dp[0][i]=1;
        }
        for(int i=0;i<m;i++)
        dp[i][0]=1;
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                dp[i][j]+=dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
```
11.Unique Path 2: https://leetcode.com/problems/unique-paths-ii/description/
```
class Solution {
public:
int dp[300][300];
    int uniquePathsWithObstacles(vector<vector<int>>& g) {

        int m=g.size();
        int n=g[0].size();
        if(g[0][0]==1||g[m-1][n-1]==1)
        return 0;
        int dp[m][n];
        memset(dp,0,sizeof(dp));
        for(int i=0;i<m;i++){
        if(g[i][0]==1)
        break;
        dp[i][0]=1;
        }
        for(int j=0;j<n;j++){
            if(g[0][j]==1)
            break;
            dp[0][j]=1;
        }
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(g[i][j])
                dp[i][j]=0;
                else
                dp[i][j]+=dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
```
12.   Minimum Path Sum : https://leetcode.com/problems/minimum-path-sum/
   Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.
```
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int n=grid.size();
        int m=grid[0].size();
        vector<vector<int>>dp(n,vector<int>(m,0));
        dp[0][0]=grid[0][0];
        for(int i=1;i<n;i++)
        dp[i][0]=dp[i-1][0]+grid[i][0];
        for(int j=1;j<m;j++)
        dp[0][j]=dp[0][j-1]+grid[0][j];
        for(int i=1;i<n;i++){
            for(int j=1;j<m;j++){
                dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j];
            }
        }
        return dp[n-1][m-1];
    }
};
```
13. Partition sum:https://leetcode.com/problems/partition-equal-subset-sum/description/
```
class Solution {
public:
int n;
bool check=false;
int x;
bool f(int pos,int sum,vector<vector<int>>&dp,vector<int>&nums){
    	if(pos==0)
    		return sum==0;
    	if(dp[pos][sum]!=-1)
    		return dp[pos][sum];
	if(sum>=nums[pos])
		return dp[pos][sum]=f(pos-1,sum-nums[pos],dp,nums)|f(pos-1,sum,dp,nums);
    	else
    		return dp[pos][sum]=f(pos-1,sum,dp,nums);
}
    bool canPartition(vector<int>& nums) {
        n=nums.size();
        int sum1=accumulate(nums.begin(),nums.end(),0);
        if(sum1%2==1)
            return false;
        int sum=sum1/2;
        vector<vector<int>>dp(n+1,vector<int>(sum+1,-1));
        x=sum;
        return f(n-1,x,dp,nums);

    }
};
```
14. Count Subset with sum K: https://www.codingninjas.com/studio/problems/number-of-subsets_3952532?source=youtube&campaign=striver_dp_videos&utm_source=youtube&utm_medium=affiliate&utm_campaign=striver_dp_videos&leftPanelTab=0
```

int n;
int dp[101][10001];
const int md=1e9+7;
int f(int pos, int sum,vector<int> &a, int k) {
	// cout<<pos<<" "<<sum<<endl;
	  if (pos == n) {
    if (sum == k)
      return 1;
    else
      return 0;
  }
  if (dp[pos][sum] != -1)
    return dp[pos][sum];
  int take  = f(pos + 1, sum + a[pos], a, k);
  int ntake = f(pos + 1, sum, a, k);

  int ans = take%md+ntake%md;
  return dp[pos][sum]=ans%md;
}

int findWays(vector<int> &arr, int k) {
	n=arr.size();
	memset(dp,-1,sizeof(dp));
  return f(0, 0,arr, k);
}

```
15.Coin Change :https://leetcode.com/problems/coin-change/description/
```
class Solution {
public:
int total;
//minimum no of coins
int dp[13][10005];
int n;
    int f(int pos,vector<int>&coins,int a){
        if(a<0)
        return INT_MAX-1;
        if(pos==n){
            if(a==0)
            return 0;
            else
            return INT_MAX-1;
        }
        if(dp[pos][a]!=-1)
        return dp[pos][a];
        if(a-coins[pos]>=0){
        int take= 1+f(pos,coins,a-coins[pos]);
        int ntake=f(pos+1,coins,a);
        return dp[pos][a]=min({take,ntake});
        }
        else{
            int ntake=f(pos+1,coins,a);
            return dp[pos][a]=ntake;
        }
    }
    int coinChange(vector<int>& coins, int amount) {
        n=coins.size();
        this->total=amount;
        memset(dp,-1,sizeof(dp));
       int x= f(0,coins,amount);
       if(x==INT_MAX-1)
       return -1;
       return x;
    }
};
```
16.Target Sum: https://leetcode.com/problems/target-sum/description/
```
class Solution {
public:
int dp[21][10001];
int n,am=0;
    int f(int pos,vector<int>&nums,int t){
        if(t<0)
        return INT_MIN;
        if(pos==n){
            if(t==am)
            return 1;
            return INT_MIN+1;
        }
        if(dp[pos][t]!=-1)
        return dp[pos][t];
        int add=max(0,f(pos+1,nums,t+nums[pos]));
        int sub=max(0,f(pos+1,nums,t-nums[pos]));
        return dp[pos][t]=add+sub;
    }
    int findTargetSumWays(vector<int>& nums, int target) {
        memset(dp,-1,sizeof(dp));
        am=target;
        am+=1000;
        n=nums.size();
        return f(0,nums,1000);
    }
};
```
17. Coin Change(2): https://leetcode.com/problems/coin-change-ii/description/
```
class Solution {
public:
int am;
int dp[301][5008];
int n;
int f(int pos,int x,vector<int>&coins){
    if(x<0)
    return INT_MIN;
    if(pos>=n){
        if(x==0)
        return 1;
        else
        return INT_MIN;
    }
    if(dp[pos][x]!=-1)
    return dp[pos][x];
    int take=max(0,f(pos,x-coins[pos],coins));
    int ntake=max(0,f(pos+1,x,coins));
    int total=take+ntake;
    return dp[pos][x]=total;
}
    int change(int amount, vector<int>& coins) {
        n=coins.size();
        memset(dp,-1,sizeof(dp));
        return f(0,amount,coins);
    }
};
```
18. Rod Cutting: https://practice.geeksforgeeks.org/problems/rod-cutting0840/1
```
class Solution{
    int n;
    int dp[1001][1001];
    int f(int pos,int len, int *price){
        if(len<0)
        return INT_MIN;
        if(pos==n){
            if(len==0)
            return 0;
            else
            return INT_MIN;
        }
        if(dp[pos][len]!=-1)
        return dp[pos][len];
        int take=f(pos,len-(pos+1),price)+price[pos];
        int ntake=f(pos+1,len,price);
        int ans=max(take,ntake);
        return dp[pos][len]=ans;
    }
  public:
    int cutRod(int price[], int n) {
        memset(dp,-1,sizeof(dp));
        this->n=n;
        return f(0,n,price);
    }
};

```
19. LCS:  (https://www.codingninjas.com/studio/problems/longest-common-subsequence_624879?source=youtube&campaign=striver_dp_videos&utm_source=youtube&utm_medium=affiliate&utm_campaign=striver_dp_videos)
```

int dp[1001][1001];

int n,m;
int f(int i,int j,string &s,string &t){
	if(j==m||i==n){
		return 0;
	}
	if(dp[i][j]!=-1)
	return dp[i][j];
	if(s[i]==t[j])
	return dp[i][j]=1+f(i+1,j+1,s,t);
	else{
		return dp[i][j]=max(f(i+1,j,s,t),f(i,j+1,s,t));
	}
}
int lcs(string s, string t)
{
	n=s.size();
	m=t.size();
	memset(dp,-1,sizeof(dp));
	return f(0,0,s,t);
}
```
20. Problem Statement:
Let there be N workers and N jobs. Any worker can be assigned to perform any job, incurring some cost that may vary depending on the work-job assignment. 
It is required to perform all jobs by assigning exactly one worker to each job and exactly one job to each agent in such a way 
that the total cost of the assignment is minimized.

Input Format:
Number of workers and job: N
Cost matrix C with dimension N*N where C(i,j) is the cost incurred on assigning ith Person to jth Job.

Sample Input:
4
9 2 7 8
6 4 3 7
5 8 1 8
7 6 9 4

Sample Output:
13

```
Constraints:
N <= 20
*/

const int N = 20;
int n, m;
ll c[N][N];
int dp[1 << 20];
ll f(ll mask) {
	int set=__builtin_popcount(mask);
	// (set) indexed job is now being investigated , that who will get it done in least amount
	if(set==n)
	return 0;
	if(dp[mask]!=-1)
	return dp[mask];
	 ll ans=INT_MAX;
	for(int j=0;j<n;j++){
	if(!(mask&(1<<j))){
	ans=min(ans,c[set][j]+f(mask|(1ll<<j)));	
	}
	}
	return dp[mask]=ans;
	}
int main() {
	  cin >> n;
    memset(dp, -1, sizeof dp);
    for(int i=0;i<n;i++){
    	for(int j=0;j<n;j++)
    	cin>>c[i][j];
    }   
    cout <<f(0LL)<<endl;
   return 0;
}
```
21.Word break:Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

 

Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Sol:
```
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        
        int n=s.size();
        set<string>st;
        sort(wordDict.begin(),wordDict.end());
        vector<int>dp(n+1,0);
        for(auto u:wordDict)
        st.insert(u);
        //dp[i] represents that wrd from 0 till i can be made
        dp[0]=1;
        for(int i=1;i<=n;i++){
            for(int j=i-1;j>=0;j--){
                if(dp[j]){
                    string word=s.substr(j,i-j);
                    
                    if(st.count(word)){
                        dp[i]=1;
                        break;
                    }
                }
            }
        }
        return dp[n];
    }
};
```
22.Rat in a maze problem :
Consider a rat placed at (0, 0) in a square matrix of order N * N. It has to reach the destination at (N - 1, N - 1). Find all possible paths that the rat can take to reach from source to destination. The directions in which the rat can move are 'U'(up), 'D'(down), 'L' (left), 'R' (right). Value 0 at a cell in the matrix represents that it is blocked and rat cannot move to it while value 1 at a cell in the matrix represents that rat can be travel through it.
Note: In a path, no cell can be visited more than one time. If the source cell is 0, the rat cannot move to any other cell.

Example 1:

Input:
N = 4
m[][] = {{1, 0, 0, 0},
         {1, 1, 0, 1}, 
         {1, 1, 0, 0},
         {0, 1, 1, 1}}
Output:
DDRDRR DRDDRR
Sol:
```

class Solution{
    public:
    int mn;
    vector<string>ans;
    int dx[4]={-1,1,0,0};
    int dy[4]={0,0,1,-1};
    vector<vector<int>>grid;
    bool isValid(int i,int j,int n,vector<vector<int>>&vis){
        if(i<0||j<0||i>=n||j>=mn||vis[i][j]||grid[i][j]==0)
        return false;
        return true;
    }
    void dfs(int i,int j,int n,string s,vector<vector<int>>&vis){
        if(i==n-1&&j==n-1){
        ans.push_back(s);
        return ;}
        for(int k=0;k<4;k++){
            int newX=i+dx[k];
            int newY=j+dy[k];
            if(!(isValid(newX,newY,n,vis)))
            continue;
            vis[newX][newY]=1;
            if(k==0)
            s+="U";
            else if(k==1)
            s+="D";
            else if(k==2)
            s+="R";
            else
            s+="L";
            dfs(newX,newY,n,s,vis);
            vis[newX][newY]=0;
            s.pop_back();
        }
    }
    vector<string> findPath(vector<vector<int>> &m, int n) {
        grid=m;
      vector<vector<int>>vis(n,vector<int>(n,0));
      mn=n;
      vis[0][0]=1;
      string s="";
      if(grid[0][0]==0||grid[n-1][n-1]==0)
      return ans;
      dfs(0,0,n,s,vis);
      return ans;
    }
};

    
```
23:We have a tree with N
 vertices numbered 1,2,…,N.
The ith
 edge (1≤i≤N−1)
 connects Vertex ui
​ and Vertex vi
​ and has a weight wi
.

For different vertices u
 and v
, let f(u,v)
 be the greatest weight of an edge contained in the shortest path from Vertex u
 to Vertex v
.

Your task is to find
∑1≤i<j≤Nf(i,j)
More formally, Please find the sum of the maximum weighted edge over all paths of the tree.

Input
The first line of input contains an integer N
 (2≤N≤105)
 — the number of nodes in the tree.

The next N−1
 lines contain 3
 space separated integers each ui
, vi
 (1≤ui​,vi​≤N)
 and wi
 (1≤wi​≤107)
 denoting an edge of weight wi
 between vertices ui
 and vi
.

Output
Print a single integer — the answer to the problem in a single line.

Examples
input
3
1 2 10
1 3 2
output
22 
input
5
1 2 4
2 3 1
1 4 6
4 5 12
output
75 
Note
In sample test case 1,

f(1,2)=10
f(1,3)=2
f(2,3)=10
Hence the answer is 10+2+10=22
.
In sample test case 2,

f(2,3)=1
f(1,2)=f(1,3)=4
f(1,4)=f(2,4)=f(3,4)=6
f(1,5)=f(2,5)=f(3,5)=f(4,5)=12
Hence the answer is 1+2∗4+3∗6+4∗12=75

Sol:
```
#include <bits/stdc++.h>
using namespace std;
#define ll long long 
#define pb push_back
const int mxn=1e5+100;
ll par[mxn];
ll sz[mxn];
ll k=1;
vector<vector<ll>>edges;
ll find(ll x){
  while(x!=par[x])
  x=find(par[x]);
  return x;
}
bool unite(ll x,ll y){
  if(find(x)==find(y))
  return true;
  x=find(x);
  y=find(y);
  k=sz[x]*sz[y];
  if(sz[x]>=sz[y]){
    par[y]=x;
    sz[x]+=sz[y];
  }
  else{
    par[x]=y;
    sz[y]+=sz[x];
  }
  return false;
}
int main() {
  int n;
  cin>>n;
  map<ll,ll>m;
  for(int i=0;i<n+1;i++){
  par[i]=i;
  sz[i]=1;}
  for(int i=0;i<n-1;i++){
    ll u,v,w;
    cin>>u>>v>>w;
    --u;
    --v;
   edges.pb({w,u,v});
  } 
  sort(edges.begin(),edges.end());
  ll ans=0;
  for(auto x:edges){
      ll u=x[1];
      ll v=x[2];
      ll weight=x[0];
      k=1;
      unite(u,v);
      // cout<<u<<" "<<v<<" "<<weight<<endl;
      ans+=weight*k;
    }
  cout<<ans<<endl;
  return 0;
  }
```
24.
Given an array of integers A of size N and an integer B.

The College library has N books. The ith book has A[i] number of pages.

You have to allocate books to B number of students so that the maximum number of pages allocated to a student is minimum.

A book will be allocated to exactly one student.
Each student has to be allocated at least one book.
Allotment should be in contiguous order, for example: A student cannot be allocated book 1 and book 3, skipping book 2.
Calculate and return that minimum possible number.

NOTE: Return -1 if a valid assignment is not possible.



Problem Constraints
1 <= N <= 105
 1 <= A[i], B <= 105

 HINT: Can you find how many number of students we need if we fix that one student can read atmost V number of pages ?
 Binary search on ans(number of pages)

```
bool f(int mid,vector<int>&a,int b){
    int ct=0,n=a.size(),sum=0;
    for(int i=0;i<n;i++){
        if(sum+a[i]<=mid){
            sum+=a[i];
            
        }
        else{
        ct++;
        sum=a[i];
        }
    }
    return ct<b;
}
int Solution::books(vector<int> &A, int B) {
    int n=A.size();
    if(n<B)
    return -1;
    int sum=0;
    for(auto u:A)
    sum+=u;
    int mx=*max_element(A.begin(),A.end());
    int low=mx,high=sum,ans=0;
    while(low<=high){
        int mid=(low+high)/2;
        if(f(mid,A,B)){
            ans=mid;
            high=mid-1;
        }
        else{
            low=mid+1;
        }
    }
    return ans;
    
}

```
25 .
Given an integer array nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once. You can return the answer in any order.

You must write an algorithm that runs in linear runtime complexity and uses only constant extra space.

 

Example 1:

Input: nums = [1,2,1,3,2,5]
Output: [3,5]
Explanation:  [5, 3] is also a valid answer.

```
class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        vector<int>ans;
        int n=nums.size(),res=0;
        for(int i=0;i<n;i++){
            res=res^nums[i];
        }
        int pos=0;//position of rightmost set bit
        for(int i=31;i>=0;i--){
            if(res&(1<<i)){
                pos=i;
                break;
            }
        }
        int a=0,b=0;
        for(int i=0;i<n;i++){
            if(nums[i]&(1<<pos)){
                a=a^nums[i];
            }
        }
         b=res^a;
        ans.push_back(a);
        ans.push_back(b);
        return ans;
    }
};
```
