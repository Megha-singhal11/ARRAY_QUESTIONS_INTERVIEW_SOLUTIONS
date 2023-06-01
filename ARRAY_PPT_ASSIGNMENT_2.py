#!/usr/bin/env python
# coding: utf-8

# ### Q1. Given an integer array nums of 2n integers, group these integers into n pairs (a1, b1), (a2, b2),..., (an, bn) such that the sum of min(ai, bi) for all i is maximized. Return the maximized sum.

# ### Example 1:
# Input: nums = [1,4,3,2]   
# Output: 4    
# Explanation: All possible pairings (ignoring the ordering of elements) are:
# 
# (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3   
# (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3    
# (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4   
# So the maximum possible sum is 4

# ### Algotithm:
# 1. Sort the given array nums in ascending order.
# 2. Initialize a variable max_sum to 0.
# 3. Iterate over the sorted array with a step size of 2, i.e., i ranging from 0 to length of nums - 1 with a step of 2.
# 4. In each iteration, add the element at index i to max_sum.
# 5. Return the value of max_sum.

# In[58]:


def arrayPairSum(nums):
    nums.sort()  # Sort the array in ascending order
    max_sum = 0
    for i in range(0, len(nums), 2):
        max_sum += nums[i]
    return max_sum

nums = [1,4,3,2]
print(arrayPairSum(nums))

In this case:
    TIME COMPLEXITY: The time complexity of this solution is O(n log n), where n is the length of the input array nums. This is             because the sort() function has a time complexity of O(n log n).
    SPACE COMPLEXITY : The space complexity is O(1) because the algorithm uses only a constant amount of extra space, regardless             of the size of the input array.
# ### Q2.Alice has n candies, where the ith candy is of type candyType[i]. Alice noticed that she started to gain weight, so she visited a doctor.The doctor advised Alice to only eat n / 2 of the candies she has (n is always even). Alice likes her candies very much, and she wants to eat the maximum number of different types of candies while still following the doctor's advice. Given the integer array candyType of length n, return the maximum number of different types of candies she can eat if she only eats n / 2 of them.
Ans 2. To find the maximum number of different types of candies Alice can eat while following the doctor's advice, we need to determine the number of unique candy types in the given array candyType. Alice can eat a maximum of n/2 candies, where n is the length of the array (always even).To solve this problem, we can follow these steps:

1. Initialize an empty set unique_candies to store the unique candy types.
2. Iterate over the elements in the candyType array.
3. Add each candy type to the unique_candies set.
4. After iterating through all the candies, calculate the minimum value between the length of unique_candies and n/2.
5. Return the minimum value calculated in the previous step.
Here's the implementation in Python:
# In[3]:


# Function to find number of candy types
def num_candyTypes(candies):
    # Declare a hashset to store candies
    s =  set()
 
    # Traverse the given array and
    # inserts element into set
    for i in range(len(candies)):
        s.add(candies[i])
 
    # Return the result
    return len(s)
 
# Function to find maximum number of
# types of candies a person can eat
def distribute_candies(candies):
    # Store the number of candies
    # allowed to eat
    allowed = len(candies)/2
 
    # Store the number of candy types
    types = num_candyTypes(candies)
 
    # Return the result
    if (types < allowed):
        print(int(types))
    else:
        print(int(allowed))
   
    # Given Input
candies = [1,1,2,2,3,3]
     
    # Function Call
distribute_candies(candies)


# In this  case:
#     TIME COMPLEXITY IS O(N)
#     SPACE COMPLEXITY IS O(N)

# ### Q3.We define a harmonious array as an array where the difference between its maximum value and its minimum value is exactly 1. Given an integer array nums, return the length of its longest harmonious subsequence among all its possible subsequences. A subsequence of an array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.

# ### Example 1: 
# Input: nums = [1,3,2,2,5,2,3,7]    
# Output: 5    
# Explanation: The longest harmonious subsequence is [3,2,2,2,3].

# ### APPROACH : HASHMAP

# ### Alogrithm: 
# In this approach, we make use of a hashmap which stores the number of times an element occurs in the array along with the element's value in the form (num:count_num)
# 1. We traverse over the numsnumsnums array and fill this mapmapmap once.
# 2. For every key of the mapmapmap considered, say key, we find out if the map contains the key+1. Such an element is found, since only such elements can be counted for the harmonic subsequence if key is considered as one of the element of the harmonic subsequence. 
# 3. The case of key−1 being in the harmonic subsequence will automatically be considered, when key−1 is encountered as the current key.
# 4. Now, whenver we find that key+1 exists in the keys of map, we determine the count of the current harmonic subsequence as countkey+countkey+1, where counti refers to the value corresponding to the key i in map, which reprents the number of times i occurs in the array nums.

# In[6]:


class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        numsLen, result = len(nums), 0
        counts = {}
        for val in nums:
            if val in counts:
                counts[val] += 1
            else:
                counts[val] = 1
            inc = val + 1
            dec = val - 1
            if dec in counts:
                result = max(result, counts[val] + counts[dec])
            if inc in counts:
                result = max(result, counts[val] + counts[inc])
        return result
arr = [1,0,0,0,1,0,0,0,0]
t=4
print(canPlaceFlowers(arr,t))
nums=[5,6,7,6,5,7,6]
print(findLHS(nums))

In this case :
    TIME COMPLEXITY IS O(N)
    SPACE COMPLEXITY IS O(N)
# ### Q4. You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in adjacent plots. Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return true if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule and false otherwise.
# 
# 
# ### Example :
# Input: flowerbed = [1,0,0,0,1], n = 1   
# Output: true

# ### Reasoning :
#   Firstly, let's understand the problem statement. We are provided with a list of flowerbeds and our task is to determine whether we can plant 'n' number of flowers in it or not. The function should return True if we can plant the flowers and False if we cannot. However, there is a rule to planting flowers known as the "no-adjacent-flowers rule". This means that we cannot plant a flower adjacent to another already planted flower.

# ### APPROACH : we use two-point start, end to record the ith empty pilot’s left and right flower position.
# 
# ### ALGORITHM:
# 1. for ith pilot, we calculate the distance to the left (i-start)and to right(end-i) and get the min distance
# 2. if min distance >=2 we can say that this pilot can be placed as a flower.
# 3. then we set ith pilot as 1, which means that this pilot has been placed and we reset the start point to i and decrement n
# 4. if n==0 which means all flowers have been placed.
# 
# 
# ## Special case (CORNER CASE/ PROBLEM MAY ARISE);
# 
# 1. If the first or last position is 0, it means that the left or right side has not been spent. At this time we set start or end to length of the array, because we need to find the minimum distance between the two sides of the empty position. At this time, the left position and the right position can seem infinitely large.
# 2. If n=0, it returns true by default.
# 3. [0],1 This case also returns true by default.

# In[51]:


def canPlaceFlowers(flowerbed,n):
        flowerbed = [0]+flowerbed+[0]
        flowers_planted = 0
        for i in range(1,len(flowerbed)-1):
            if (flowerbed[i-1]==0 and flowerbed[i]==0 and flowerbed[i+1]==0):
                flowerbed[i]=1
                flowers_planted+=1
            if flowers_planted>=n:
                return True
        return False
    
    
flowerbed = [1,0,0,0,1,0,0,0,0]
n = 1
print(canPlaceFlowers(flowerbed,n))

In this case: 
   Time Complexity: The time complexity is linear with respect to the input size i.e. O(n)
   Space Complexity: The only extra space used by this algorithm are two integer variables, 'flowers_planted' and 'i', which are used to keep track of the number of planted flowers and the current position being checked respectively. These variables occupy a constant amount of memory regardless of the size of the input 'flowerbed' and hence the space complexity is O(1).
# ### Q5. Given an integer array nums, find three numbers whose product is maximum and return the maximum product.

# ### Example 1:
# Input: nums = [1,2,3]    
# Output: 6

# ### Naive Approach

# ### Algorithm:
# Follow the below steps to solve the problem:
# 
# 1. Run a nested for loop to generate every subarray
# 2. Calculate the product of elements in the current subarray
# 3. Return the maximum of these products calculated from the subarrays
# Below is the implementation of the above approach:

# ### CORNER CASES:
# Problem will occur when our array will contain odd no. of negative elements. In that case, we have to reject anyone negative element so that we can even no. of negative elements and their product can be positive. Now since we are considering subarray so we can’t simply reject any one negative element. We have to either reject the first negative element or the last negative element.
# 
# But if we will traverse from starting then only the last negative element can be rejected and if we traverse from the last then the first negative element can be rejected. So we will traverse from both the end and from both the traversal we will take answer from that traversal only which will give maximum product subarray.
# 
# So actually we will reject that negative element whose rejection will give us the maximum product’s subarray.

# In[49]:


# Python3 program to find Maximum Product Subarray
 
# Returns the product of max product subarray.
 
def maxSubarrayProduct(arr,n):
 
    # Initializing result
    result = arr[0]
 
    for i in range(n):
 
        mul = arr[i]
 
        # traversing in current subarray
        for j in range(i + 1, n):
 
            # updating result every time
            # to keep an eye over the maximum product
            result = max(result, mul)
            mul *= arr[j]
 
        # updating the result for (n-1)th index.
        result = max(result, mul)
 
    return result
 
 
# EXAMPLE
arr  = [1,2,3]
n = len(arr)
print(maxSubarrayProduct(arr,n))

In this case:
    TIME COMPLEXITY : O(N^2)
    SPACE COMPLEXITY : O(1)
# ### SECOND APPROACH:   Using traversal from starting and end of an array
# ( REDUCES TIME COMPLEXITY)

# In[50]:


# Python program to find Maximum Product Subarray
 
import sys
 
# Returns the product of max product subarray.
def maxSubarrayProduct(arr, n):
    ans = -sys.maxsize - 1  # Initialize the answer to the minimum possible value
    product = 1
 
    for i in range(n):
        product *= arr[i]
        ans = max(ans, product)  # Update the answer with the maximum of the current answer and product
        if arr[i] == 0:
            product = 1  # Reset the product to 1 if the current element is 0
 
    product = 1
 
    for i in range(n - 1, -1, -1):
        product *= arr[i]
        ans = max(ans, product)
        if arr[i] == 0:
            product = 1
 
    return ans
 
# Driver code
arr = [1,2,3]
n = len(arr)
print(maxSubarrayProduct(arr,n))


# In this case:
# TIME COMPLEXITY : O(N)
# SPACE COMPLEXITY : O(1)

# ### Question 6
# Given an array of integers nums which is sorted in ascending order, and an integer target,
# write a function to search target in nums. If target exists, then return its index. Otherwise,
# return -1.
# 
# You must write an algorithm with O(log n) runtime complexity.
# 
# ### EXAMPLE:
# Input: nums = [-1,0,3,5,9,12]    
# target = 9
# Output: 4    
# Explanation: 9 exists in nums and its index is 4

# ### APPROACH: BINARY SEARCH 

# ### Algorithm:
# 1. The function search(nums, target) takes the sorted array nums and the target integer as inputs.
# 2. It initializes two pointers, left and right, to the start and end of the array respectively.
# 3. The algorithm enters a while loop that continues as long as the left pointer is less than or equal to the right pointer.
# 4. Inside the loop, it calculates the middle index mid using the formula (left + right) // 2.
# 5. It compares the value at the middle index nums[mid] with the target value.
# 6. If nums[mid] is equal to the target, the target is found, and the function returns the index mid.
# 7. If nums[mid] is less than the target, the target is in the right half of the remaining array. So, the left pointer is updated to mid + 1 to search in the right half.
# 8. If nums[mid] is greater than the target, the target is in the left half of the remaining array. So, the right pointer is updated to mid - 1 to search in the left half.
# 9. If the while loop ends without finding the target, the function returns -1 to indicate that the target is not present in the array.

# In[53]:


def search(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Example usage
nums = [-1, 0, 3, 5, 9, 12]
target = 9
result = search(nums, target)
print(result)

In this case:
    TIME COMPLEXITY IS O(logn)
    SPACE COMPLEXITY IS O(1)
# ### Q7. An array is monotonic if it is either monotone increasing or monotone decreasing. An array nums is monotone increasing if for all i <= j, nums[i] <= nums[j]. An array nums is monotone decreasing if for all i <= j, nums[i] >= nums[j].
# Given an integer array nums, return true if the given array is monotonic, or false otherwise.
# 
# ### Example :
# Input: nums = [1,2,2,3]   
# Output: true   

# ### Approach: The problem can be solved by checking if the array is in increasing order or in decreasing order. This can be easily done in the following way:
# 
# 1. If for each i in range [0, N-2], arr[i] ≥  arr[i+1] the array is in decreasing order.
# 2. If for each i in range [0, N-2], arr[i] ≤ arr[i+1], the array is in increasing order.
# ### Follow the below steps to solve the problem:
#               
#               ### ALGORITHM:
# 
# 1. Traverse the array arr[] from i = 0 to N-2 and check if the array is increasing in order
# 2. Traverse the array arr[] from i = 0 to N-2 and check if the array is decreasing in order
# 3. If neither of the above two is true, then the array is not monotonic.
# Below is the implementation of the above approach:

# In[55]:


# Python program for above approach
 
# Function to check array is monotonic
def check(arr):
    N = len(arr)
    inc = True
    dec = True
     
    # Loop to check if array is increasing
    for i in range(0, N-1):
       
        # To check if array is not increasing
        if arr[i] > arr[i+1]:
            inc = False
 
    # Loop to check if array is decreasing
    for i in range(0, N-1):
       
       # To check if array is not decreasing
        if arr[i] < arr[i+1]:
            dec = False
 
    # Pick one whether inc or dec
    return inc or dec
 
    # Function call
    ans = check(arr)
    if ans == True:
        print("Yes")
    else:
        print("No")

arr = [1,2,2,3]
print(check(arr))


# ### ANOTHER WAY (PYTHON CODE)

# In[56]:


def isMonotonic(nums):
    is_increasing = True
    is_decreasing = True

    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            is_increasing = False
        if nums[i] < nums[i + 1]:
            is_decreasing = False

    return is_increasing or is_decreasing

arr= [1,2,2,3]
print(isMonotonic(arr))

In this case:
    Time Complexity: O(N)
    Auxiliary Space: O(1)
# ### Q8.You are given an integer array nums and an integer k. In one operation, you can choose any index i where 0 <= i < nums.length and change nums[i] to nums[i] + x where x is an integer from the range [-k, k]. You can apply this operation at most once for each index i. The score of nums is the difference between the maximum and minimum elements in nums. Return the minimum score of nums after applying the mentioned operation at most once for each index in it.
# 
# ### EXAMPLE:
# Input: nums = [1]    
# k = 0    
# Output: 0    
# Explanation: The score is max(nums) - min(nums) = 1 - 1 = 0.

# ### ALGORITHM:
# 1. Find the minimum and maximum values in the array nums and calculate the initial score as max(nums) - min(nums).
# 2. Iterate through each element num in nums.
# 3. For each num, calculate the minimum and maximum values that can be obtained by applying the operation num + x where x is in the range [-k, k]. These values can be calculated as min_val = min(num - k, current_min) and max_val = max(num + k, current_max), where current_min and current_max are the current minimum and maximum values obtained so far.
# 4. Update the current minimum and maximum values if the newly calculated values are smaller or larger, respectively.
# 5. After iterating through all elements, calculate the final score as current_max - current_min and return it.   
# Here's the Python code that implements this algorithm:

# In[57]:


def minimum_score(nums, k):
    min_val = float('inf')
    max_val = float('-inf')

    for num in nums:
        min_val = min(num - k, min_val)
        max_val = max(num + k, max_val)

    return max_val - min_val

# Example usage
nums =[1]
k = 0

result = minimum_score(nums, k)
print(result)

In this case:
TIME COMPLEXITY : O(N)- the algorithm iterates through each element in nums exactly once, performing a constant amount of operations for each element.
SPACE COMPLEXITY: O(1)-it uses a constant amount of extra space that does not depend on the size of the input.