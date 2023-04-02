# DP-Series-
Sheet 
1. Partition equal sum : https://leetcode.com/problems/partition-equal-subset-sum/description/
```
// This is my first code on Github
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
        for(int i=0;i<=n;i++)
            dp[i][0]=1;
        for(int i=0;i<=sum;i++)
            dp[0][i]=0;
        for(int i=1;i<=n;i++)
        {
            for(int j=1;j<=sum;j++)
            {
                dp[i][j]=dp[i-1][j];
                if(j>nums[i-1])
                {
                  bool x=dp[i-1][j];
                  bool y=dp[i][j-nums[i-1]];
                  dp[i][j]=x|y;
              }
          }
      }
      return dp[n][sum];
    }
};
```
