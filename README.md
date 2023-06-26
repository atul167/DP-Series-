# DP-Series-
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
