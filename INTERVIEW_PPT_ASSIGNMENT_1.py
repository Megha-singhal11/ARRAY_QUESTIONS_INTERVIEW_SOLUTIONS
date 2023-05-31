#!/usr/bin/env python
# coding: utf-8

# ## Q1.Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.
# 
# ## Example:
# Input: nums = [2,7,11,15], target = 9
# Output0 [0,1]
# 
# Explanation: Because nums[0] + nums[1] == 9, we return [0, 1]

# SOLUTION - In order to solve tis question, we will use two approaches :

# ## FIRST APPROACH (BRUTE FORCE)

# In[1]:


def twoSum(nums, target):
    # Create a dictionary to store the complement of each number
    complement_dict = {}

    # Iterate through the array
    for i, num in enumerate(nums):
        complement = target - num
        # Check if the complement exists in the dictionary
        if complement in complement_dict:
            # Return the indices of the current number and its complement
            return [complement_dict[complement], i]
        else:
            # Add the current number and its index to the dictionary
            complement_dict[num] = i

    # No two numbers found that add up to the target
    return []

# Example usage
nums = [2, 7, 11, 15]
target = 9
result = twoSum(nums, target)
print(result) 

    IN THIS CASE :
TIME COMPLEXITY IS O(N^2)
SPACE COMPLEXITY IS O(1)
# ## SECOND APPROACH (WE CAN REDUCE TIME COMPLEXITY)
# ### HASH MAP

# In[2]:


class solution :
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        hash_map = {}
        
        for i in range(len(nums)):
            if nums[i] in hash_map:
                return [i, hash_map[nums[i]]]
            else:
                hash_map[target - nums[i]] = i
        return       

nums : list[2, 7, 11, 15]
target = 9
result = twoSum(nums, target)
print(result) 

IN THIS CASE :
    TIME COMPLEXITY IS O(N)
    SPACE COMPLEXITY IS O(N)
# ### Q2.Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val. Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things: 
# 
#     Change the array nums such that the first k elements of nums contain the elements which are not equal to val. 
#     The remaining elements of nums are not important as well as the size of nums. Return k.
# 
# ### Example : Input: nums = [3,2,2,3], val = 3 Output: 2, nums = [2,2,*,*]
# 
# Explanation: Your function should return k = 2, with the first two elements of nums being 2. It does not matter what you leave beyond the returned k (hence they are underscores)

# ## Approach: two pointers appraoch
# To solve the problem and meet the given requirements, we can use a two-pointer approach.

Algorithm:
1. We have two pointers, i and k, where i iterates through the array and k keeps track of the current position for elements that are not equal to val.

2. We iterate through the array and check if the current element is not equal to val.

3. If it's not equal to val, we move it to the kth position and increment k.

4. After processing all elements, the modified array will have the desired elements in the first k positions.

5. Finally, We return the value of k, which represents the count of elements not equal to val.

Here's an implementation that modifies the array in-place and returns the count of elements that are not equal to the given value:
# In[3]:


def removeElement(nums, val):
    # Initialize two pointers
    i = 0  # pointer to the current position
    k = 0  # count of elements not equal to val

    # Iterate through the array
    while i < len(nums):
        # If the current element is not equal to val,
        # move it to the kth position and increment k
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1
        i += 1

    # Return the count of elements not equal to val (k)
    return k

# Example usage
nums = [3, 2, 2, 3]
val = 3
k = removeElement(nums, val)
print(k)       
print(nums) 

IN THIS CASE :
    TIME COMPLEXITY IS O(N)
    SPACE COMPLEXITY IS O(N)
# ### Q3.Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order. You must write an algorithm with O(log n) runtime complexity.
# 
# ### Example 1: Input: nums = [1,3,5,6], target = 5
## To solve this problem with a runtime complexity of O(log n), we can use the binary search algorithm to find the target or the position where it should be inserted.Algorithm:
Initialize two pointers, left and right, to the start and end of the array, respectively.

Enter a while loop that continues as long as left is less than or equal to right.

Calculate the middle index mid using integer division.

Check if the value at index mid is equal to the target. If it is, return mid.

If the value at index mid is less than the target, update left to mid + 1 to search in the right half of the array.

If the value at index mid is greater than the target, update right to mid - 1 to search in the left half of the array.

Repeat steps 3-6 until the target is found or the pointers cross each other.

If the target is not found, return the value of left as it represents the position where the target would be inserted.

Here's an implementation in Python:
# In[4]:


def searchInsert(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return left

# Example usage
nums = [1, 3, 5, 6]
target = 5
index = searchInsert(nums, target)
print(index)

IN THIS CASE :
    TIME COMPLEXITY IS O(logn)
    SPACE COMPLEXITY IS O(1)
# ### Q4. You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's. Increment the large integer by one and return the resulting array of digits.
 To solve this problem, you can follow a simple algorithm to increment the given array of digits as if it represents a large integer. Here's an implementation in Python:
# In[5]:


def plusOne(digits):
    n = len(digits)
    carry = 1

    # decending order sorting
    for i in range(n - 1, -1, -1):
        # Add the carry to the current digit
        digits[i] += carry
        carry = digits[i] // 10
        digits[i] %= 10

        # no carry, results in returing the digits
        if carry == 0:
            return digits

    # If still remains a carry, we need to insert a new digit at the beginning
    if carry == 1:
        digits.insert(0, carry)

    return digits

# Example usage
digits = [1, 2, 3]
result = plusOne(digits)
print(result)

Time complexity -O(n)-This is because we traverse the array once in the worst case, incrementing each digit by 1 and handling any resulting carries.
space complexity -O(1)-because we are modifying the input array in-place without using any additional data structures that scale with the input size.
# ### Q5. You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.Merge nums1 and nums2 into a single array sorted in non-decreasing order. The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

# ### Example 1: Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3 Output: [1,2,2,3,5,6]

# ## Approach -Two pointers approach
# To merge two sorted arrays, nums1 and nums2, into nums1 in non-decreasing order, you can use a two-pointer approach. Since nums1 has enough space to accommodate both arrays, you can start comparing elements from the end of each array and place them in the correct positions in nums1.
Algorithm: 
    1. initialise a new nums1 array containing the first m elements of num1.
    2. initialise p1 to begning of nums/copy
    3. initialise p2 to begning of nums2.
    4. if nums/copy[p1] exists and is less than or equal to nums2[p2]
              print nums/copy[p1] in num1 and increment p1
        else 
             print nums2[p2] in nums1 and increment p2.
# In[6]:


def merge(nums1, m, nums2, n):
    p1 = m - 1  # Pointer for nums1
    p2 = n - 1  # Pointer for nums2
    p = m + n - 1 
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] >= nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1

    # If there are remaining elements in nums2, copy them to nums1
    nums1[:p2 + 1] = nums2[:p2 + 1]

# Example usage
nums1 = [1, 2, 3, 0, 0, 0]
m = 3
nums2 = [2, 5, 6]
n = 3

merge(nums1, m, nums2, n)
print(nums1)    

In this case:
    Time Complexity : O(nlogn)
    Space Complexity : O(log n)
# ### Q6. Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
# 
# ### Example 1: Input: nums = [1,2,3,1] , output True

# In[7]:


def containsDuplicate(nums) -> bool:
        # Sort the list first
        nums.sort()

        i = 0
        while i < len(nums) - 1:
            currNum = nums[i]
            nextNum = nums[i+1]
            if currNum == nextNum:
                return True

            i += 1

        return False
nums = [1,2,3,1]
result = containsDuplicate(nums)
print(result)

Time complexity: O(nlogn). Sorting is O(nlogn) in the worst case scenario and iterating through the list with 2 pointers is O(n). Therefore, the entire algorithm is dominated by the sorting step, which is O(nlogn).

Space complexity: O(1). Space depends on the sorting implementation which usually costs O(1) extra space if Heap Sort is used. Heap Sort is an in-place designed sorting algorithm, the space requirement is constant and therefore, O(1).
# ### Q7. Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the nonzero elements.

# ### Approach -Two pointers approach
# To move all zeros to the end of the given array nums while maintaining the relative order of nonzero elements, you can use a two-pointer approach.

# ### Example :
# Input: nums = [0,1,0,3,12]    
# Output: [1,3,12,0,0]

# In[8]:


def move_zeros(nums: list) -> list:
    # 
    size = len(nums)
    left = right = 0
    while right < size:
        if nums[right] == 0:
            right += 1
        else:
            nums[left], nums[right] = nums[right], nums[left]
            right += 1
            left += 1
    return nums

 
nums = [0,1,0,3,1,2]
print(move_zeros(nums))

In this case:

The time complexity of the provided solution is O(n), where n is the length of the input array nums. The solution involves iterating through the array once with two pointers, left and right. Each element is visited and potentially swapped once, so the time complexity is linear with respect to the size of the input array.

The space complexity of the solution is O(1) because it uses only a constant amount of extra space to store the two pointers and temporary variables for swapping elements. The space required does not depend on the size of the input array, thus making it an in-place solution.
# ### Q8.You have a set of integers s, which originally contains all the numbers from 1 to n. Unfortunately, due to some error, one of the numbers in s got duplicated to another number in the set, which results in repetition of one number and loss of another number.
# 
# You are given an integer array nums representing the data status of this set after the error.
# 
# Find the number that occurs twice and the number that is missing and return them in the form of an array.

# ### Example :
# Input: nums = [1,2,2,4]   
# Output: [2,3]

# ### Algorithm:
# To find the missing number, we use the formula for the sum of consecutive integers from 1 to n, which is (n * (n + 1) // 2). We subtract the sum of num_set from this expected sum to obtain the missing number.

# In[9]:


def findErrorNums(nums):
    n = len(nums)
    num_set = set(nums)
    duplicate = sum(nums) - sum(num_set)
    missing = (n * (n + 1) // 2) - sum(num_set)
    return [duplicate, missing]

# Example usage
nums = [1, 2, 2, 4]
result = findErrorNums(nums)
print(result)  


# The Time complexity of the provided solution is O(n), where n is the length of the input array nums. The solution involves iterating through the array once to calculate the sum of all elements and the sum of the elements in the set.
# 
# The Space complexity of the solution is O(n) as well. The set num_set can potentially store all unique elements from the array, which would require O(n) space in the worst case. Additionally, the solution uses a constant amount of extra space to store the duplicate and missing numbers in the output array.
