import numpy as np
import pandas as pd

# تابع دیفازی‌سازی برای تبدیل عدد فازی مثلثی به عدد قطعی
def defuzzify(triangular_number):
    a, b, c = triangular_number
    crisp_value = (a + b + c) / 3  # مرکز سطح (Center of Area)
    return crisp_value

# 1. ساخت ماتریس مقایسه زوجی از فایل CSV با اعداد فازی مثلثی
def load_pairwise_comparison(file_path, num_criteria):
    df = pd.read_csv(file_path)
    PW = np.ones((num_criteria, num_criteria), dtype=object)  # ذخیره به عنوان اشیاء فازی
    for _, row in df.iterrows():
        i = int(row['Criteria 1']) - 1
        j = int(row['Criteria 2']) - 1
        triangular_number = (row['Comparison Value(a)'], row['Comparison Value(b)'], row['Comparison Value(c)'])
        PW[i, j] = triangular_number
        PW[j, i] = (1 / triangular_number[2], 1 / triangular_number[1], 1 / triangular_number[0])
    return PW

# 2. محاسبه وزن‌های فازی AHP از ماتریس مقایسه زوجی
def calculate_fuzzy_weights(PW):
    num_criteria = PW.shape[0]
    col_sum = [np.sum([PW[row, col] for row in range(num_criteria)], axis=0) for col in range(num_criteria)]
    normalized_PW = [[(PW[row, col][0] / col_sum[col][0], PW[row, col][1] / col_sum[col][1], PW[row, col][2] / col_sum[col][2])
                      for col in range(num_criteria)] for row in range(num_criteria)]
    fuzzy_weights = [tuple(np.mean([normalized_PW[row][col] for col in range(num_criteria)], axis=0)) for row in range(num_criteria)]
    return fuzzy_weights

# 3. دیفازی‌سازی وزن‌های فازی
def defuzzify_weights(fuzzy_weights):
    crisp_weights = [defuzzify(weight) for weight in fuzzy_weights]
    return crisp_weights

# 4. محاسبه نسبت سازگاری
def calculate_consistency_ratio(PW, weights, random_index):
    num_criteria = PW.shape[0]
    weighted_PW = np.zeros((num_criteria, num_criteria))
    for col in range(num_criteria):
        for row in range(num_criteria):
            weighted_PW[row, col] = weights[col] * defuzzify(PW[row, col])

    weighted_sum = np.zeros(num_criteria)
    for row in range(num_criteria):
        weighted_sum[row] = np.sum(weighted_PW[row, :])

    lambda_max = np.sum(weighted_sum / weights) / num_criteria
    consistency_index = (lambda_max - num_criteria) / (num_criteria - 1)
    consistency_ratio = consistency_index / random_index[num_criteria]
    return consistency_ratio

# 5. ساخت ماتریس تصمیم از فایل CSV
def load_decision_matrix(file_path, num_options, num_criteria):
    df = pd.read_csv(file_path)
    D = np.zeros((num_options, num_criteria))
    for _, row in df.iterrows():
        i = int(row['Option']) - 1
        j = int(row['Criteria']) - 1
        D[i, j] = row['Value']
    return D

# 6. ساخت ماتریس تصمیم وزن‌دهی‌شده و تعیین ایده‌آل‌ها
def weighted_decision_matrix(D, weights, best_direction):
    num_options, num_criteria = D.shape
    U = np.sqrt(np.sum(D ** 2, axis=0))
    D_weighted = D * (weights / (U + 1e-10))  # اضافه کردن مقدار کوچک برای جلوگیری از تقسیم بر صفر
    Ib = np.zeros(num_criteria)
    Iw = np.zeros(num_criteria)
    for col in range(num_criteria):
        if best_direction[col] == 1:
            Ib[col] = np.max(D_weighted[:, col])
            Iw[col] = np.min(D_weighted[:, col])
        else:
            Ib[col] = np.min(D_weighted[:, col])
            Iw[col] = np.max(D_weighted[:, col])
    return D_weighted, Ib, Iw

# 7. بهینه‌سازی وزن‌ها با استفاده از PSO
def pso_optimize_weights(initial_weights, D_weighted, Ib, num_particles=30, max_iter=100, c1=1.5, c2=1.5, w=0.5):
    num_criteria = len(initial_weights)
    particles = np.random.rand(num_particles, num_criteria)  # مقداردهی اولیه ذرات
    velocities = np.zeros_like(particles)  # سرعت‌ها
    pbest = particles.copy()  # بهترین موقعیت ذرات
    gbest = particles[0]  # بهترین موقعیت کلی

    def objective_function(weights):
        D_opt = D_weighted * weights / (np.sqrt(np.sum((D_weighted * weights) ** 2, axis=0)) + 1e-10)
        return np.linalg.norm(D_opt - Ib)

    best_score = objective_function(gbest)
    for i in range(num_particles):
        score = objective_function(pbest[i])
        if score < best_score:
            gbest = pbest[i]
            best_score = score

    for iter in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0.01, 1)
            particles[i] /= np.sum(particles[i])
            score = objective_function(particles[i])
            if score < objective_function(pbest[i]):
                pbest[i] = particles[i]
                if score < best_score:
                    gbest = particles[i]
                    best_score = score
    return gbest

# 8. محاسبه بهترین انتخاب با استفاده از TOPSIS
def calculate_best_choice(D_weighted, Ib, Iw):
    num_options, num_criteria = D_weighted.shape
    distance_to_Ib = np.sqrt(np.sum((Ib - D_weighted) ** 2, axis=1))
    distance_to_Iw = np.sqrt(np.sum((Iw - D_weighted) ** 2, axis=1))
    choices = distance_to_Iw / (distance_to_Ib + distance_to_Iw + 1e-10)  # اضافه کردن مقدار کوچک به مخرج
    best_choice_index = np.argmax(choices)
    best_choice = choices[best_choice_index]
    return best_choice, best_choice_index + 1

# فایل‌های ورودی و مشخصات داده‌ها
pairwise_file = r'C:\Users\RayaBit\Desktop\Smart_Transportation\main.csv'
decision_file = r'C:\Users\RayaBit\Desktop\Smart_Transportation\article_dataset_customize2.csv'
num_criteria = 3
num_options = 4
random_index = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
best_direction = [0, 0, 0]

# اجرا
PW = load_pairwise_comparison(pairwise_file, num_criteria)
fuzzy_weights = calculate_fuzzy_weights(PW)
weights = defuzzify_weights(fuzzy_weights)
consistency_ratio = calculate_consistency_ratio(PW, weights, random_index)
D_original = load_decision_matrix(decision_file, num_options, num_criteria)

optimized_weights_list = []
scores_list = []
best_choices_count = {}

for _ in range(30):
    # بازیابی ماتریس تصمیم به حالت اولیه
    D = D_original.copy()
    # اعمال وزن‌های اولیه AHP
    D_weighted, Ib, Iw = weighted_decision_matrix(D, weights, best_direction)

    # اجرای PSO برای بهینه‌سازی وزن‌ها
    optimized_weights = pso_optimize_weights(weights, D_weighted, Ib)
    optimized_weights_list.append(optimized_weights)

    # بازیابی ماتریس تصمیم به حالت اولیه
    D = D_original.copy()

    # محاسبه ماتریس وزن‌دهی شده جدید و ایده‌آل‌ها برای بهینه‌ترین وزن‌ها
    D_weighted, Ib, Iw = weighted_decision_matrix(D, optimized_weights, best_direction)
    best_choice, best_choice_index = calculate_best_choice(D_weighted, Ib, Iw)
    scores_list.append(best_choice)

    if best_choice_index in best_choices_count:
        best_choices_count[best_choice_index] += 1
    else:
        best_choices_count[best_choice_index] = 1

average_score = np.mean(scores_list)
average_weights = np.mean(optimized_weights_list, axis=0)
# ترکیب معیارها برای انتخاب بهترین مسیر (ترکیب تعداد دفعات و میانگین امتیاز)
combined_scores = {}
for choice_index in best_choices_count:
    count = best_choices_count[choice_index]
    # امتیاز نهایی به صورت ترکیب میانگین امتیاز و تعداد دفعات
    combined_scores[choice_index] = 0.5 * count + 0.5 * average_score

# انتخاب بهترین مسیر بر اساس امتیاز ترکیبی
most_frequent_choice = max(combined_scores, key=combined_scores.get)
total_combined_score = sum(combined_scores.values())

print("Average Optimized Weights:", average_weights)
print("Average Score:", average_score)
print(f"Most Frequent Choice is Option {most_frequent_choice} with combined score: {combined_scores[most_frequent_choice]/total_combined_score}")

