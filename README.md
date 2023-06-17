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
