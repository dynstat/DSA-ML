use std::collections::HashMap;

pub fn subarrays_div_by_k(nums: Vec<i32>, k: i32) -> i32 {
    let mut mod_count = HashMap::new();
    mod_count.insert(0, 1);
    let mut prefix_sum = 0;
    let mut result = 0;

    for num in nums {
        prefix_sum += num;
        let mut modulo = prefix_sum % k;
        if modulo < 0 {
            modulo += k;
        }

        result += *mod_count.get(&modulo).unwrap_or(&0);
        *mod_count.entry(modulo).or_insert(0) += 1;
    }

    result
}

fn main() {
    let nums1 = vec![4, 5, 0, -2, -3, 1];
    let k1 = 5;
    let result1 = subarrays_div_by_k(nums1, k1);
    println!("For nums1 and k1={}, the result is: {}", k1, result1);

    let nums2 = vec![5];
    let k2 = 9;
    let result2 = subarrays_div_by_k(nums2, k2);
    println!("For nums2 and k2={}, the result is: {}", k2, result2);
}