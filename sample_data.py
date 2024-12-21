import pandas as pd
from sklearn.datasets import fetch_california_housing

def generate_large_csv(file_name='sample_data_large.csv', n_samples=10000):
    # 加载 California Housing 数据集
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    # 检查是否需要增加样本量
    current_samples = df.shape[0]
    if n_samples > current_samples:
        # 计算需要复制多少次
        reps = n_samples // current_samples
        remainder = n_samples % current_samples

        # 复制数据
        df_large = pd.concat([df] * reps, ignore_index=True)

        # 添加剩余的数据
        if remainder > 0:
            df_large = pd.concat([df_large, df.sample(n=remainder, random_state=42)], ignore_index=True)
    else:
        # 如果不需要增加样本量，随机采样 n_samples 条记录
        df_large = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # 保存为 CSV 文件
    df_large.to_csv(file_name, index=False)
    print(f"{file_name} 文件已成功生成，包含 {df_large.shape[0]} 条记录！")

if __name__ == "__main__":
    generate_large_csv()
