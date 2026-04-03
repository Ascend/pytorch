import argparse
from collections import defaultdict
from pathlib import Path
import re
import pandas as pd


def extract_log_info(log_file, profile_dir='./profile', output_file='log_analysis.xlsx'):
    """
    提取日志文件中各模型的训练时间信息和profile数据
    
    Args:
        log_file: 日志文件路径
        profile_dir: profile数据目录路径
        output_file: 输出Excel文件路径
    """
    # 模式匹配
    # 修改1: 同时支持npu和cuda
    model_pattern = r'(npu|cuda)\s+train\s+(\S+)'
    eager_pattern = r'eager.*avg step time:\s*([\d\.]+)\s*ms'
    compile_pattern = r'compile.*avg step time:\s*([\d\.]+)\s*ms'
    
    # 模式匹配：算子编译时间
    op_compile_time_pattern = r'op_compile_time:\s*([\d\.]+)\s*ms'
    
    # 存储结果
    data = defaultdict(lambda: {
        'accuracy': None,  # 修改2: 存储完整的精度校验日志
        'eager_E2E_avg_time': None, 
        'compile_E2E_avg_time': None,
        'op_compile_time': None,
        'eager_OP_avg_time': None,
        'compile_OP_avg_time': None
    })
    current_model = None
    in_compile_block = False
    compile_block_lines = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for _, line in enumerate(lines):
        # 匹配模型名
        model_match = re.search(model_pattern, line)
        if model_match:
            # 处理上一个模型的compile块日志
            if current_model and in_compile_block and compile_block_lines:
                # 提取pass_accuracy日志
                for log_line in compile_block_lines:
                    if 'pass_accuracy' in log_line:
                        data[current_model]['accuracy'] = log_line.strip()
                        break
                
                # 提取op_compile_time
                for log_line in compile_block_lines:
                    op_compile_match = re.search(op_compile_time_pattern, log_line)
                    if op_compile_match:
                        data[current_model]['op_compile_time'] = float(op_compile_match.group(1))
                        break
            
            # 重置状态
            current_model = model_match.group(2)  # 第二个分组是模型名
            in_compile_block = False
            compile_block_lines = []
            continue
        
        # 匹配eager模式时间
        if current_model:
            eager_match = re.search(eager_pattern, line)
            if eager_match:
                data[current_model]['eager_E2E_avg_time'] = float(eager_match.group(1))
            
            # 匹配compile模式时间
            compile_match = re.search(compile_pattern, line)
            if compile_match:
                data[current_model]['compile_E2E_avg_time'] = float(compile_match.group(1))
                in_compile_block = True
                compile_block_lines = []  # 开始收集compile块日志
        
        # 收集compile块的日志
        if current_model and in_compile_block:
            compile_block_lines.append(line)
    
    # 处理最后一个模型的compile块日志
    if current_model and in_compile_block and compile_block_lines:
        for log_line in compile_block_lines:
            if 'pass_accuracy' in log_line:
                data[current_model]['accuracy'] = log_line.strip()
                break
        
        for log_line in compile_block_lines:
            op_compile_match = re.search(op_compile_time_pattern, log_line)
            if op_compile_match:
                data[current_model]['op_compile_time'] = float(op_compile_match.group(1))
                break
    
    # 需求3: 读取profile目录中的step_trace_time.csv文件
    profile_path = Path(profile_dir)
    if profile_path.exists():
        for model_dir in profile_path.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                
                # 读取eager模式下的step_trace_time.csv
                # 修改3: 自动获取下一级目录
                eager_dir = model_dir / 'eager'
                if eager_dir.exists() and eager_dir.is_dir():
                    # 获取eager目录下的第一个子目录
                    eager_subdirs = list(eager_dir.iterdir())
                    if eager_subdirs:
                        eager_subdir = eager_subdirs[0]  # 假设只有一个子目录
                        eager_csv_path = eager_subdir / 'ASCEND_PROFILER_OUTPUT' / 'step_trace_time.csv'
                        if eager_csv_path.exists():
                            try:
                                eager_df = pd.read_csv(eager_csv_path, sep=',')
                                if 'Computing' in eager_df.columns:
                                    data[model_name]['eager_OP_avg_time'] = eager_df['Computing'].mean() / 1e3
                                else:
                                    print(f"警告: {eager_csv_path} 中没有Computing列")
                            except Exception as e:
                                print(f"读取{eager_csv_path}时出错: {e}")
                
                # 读取compile模式下的step_trace_time.csv
                compile_dir = model_dir / 'compile'
                if compile_dir.exists() and compile_dir.is_dir():
                    # 获取compile目录下的第一个子目录
                    compile_subdirs = list(compile_dir.iterdir())
                    if compile_subdirs:
                        compile_subdir = compile_subdirs[0]  # 假设只有一个子目录
                        compile_csv_path = compile_subdir / 'ASCEND_PROFILER_OUTPUT' / 'step_trace_time.csv'
                        if compile_csv_path.exists():
                            try:
                                compile_df = pd.read_csv(compile_csv_path, sep=',')
                                if 'Computing' in compile_df.columns:
                                    data[model_name]['compile_OP_avg_time'] = compile_df['Computing'].mean() / 1e3
                                else:
                                    print(f"警告: {compile_csv_path} 中没有Computing列")
                            except Exception as e:
                                print(f"读取{compile_csv_path}时出错: {e}")
    else:
        print(f"警告: profile目录不存在: {profile_dir}")
    
    # 转换为DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'model_name'
    df.reset_index(inplace=True)

    # 计算速度提升率
    # 1. 计算E2E_speed_up_rate = eager_E2E_avg_time / compile_E2E_avg_time
    # 2. 计算OP_speed_up_rate = eager_OP_avg_time / compile_OP_avg_time
    df['E2E_speed_up_rate'] = df.apply(
        lambda row: row['eager_E2E_avg_time'] / row['compile_E2E_avg_time'] 
        if row['compile_E2E_avg_time'] and row['compile_E2E_avg_time'] != 0 else None, 
        axis=1
    )
    
    df['OP_speed_up_rate'] = df.apply(
        lambda row: row['eager_OP_avg_time'] / row['compile_OP_avg_time'] 
        if row['compile_OP_avg_time'] and row['compile_OP_avg_time'] != 0 else None, 
        axis=1
    )
    
    # 重排列顺序，使相关列更清晰
    column_order = [
        'model_name', 'accuracy', 'op_compile_time',
        'eager_E2E_avg_time', 'compile_E2E_avg_time', 'E2E_speed_up_rate',
        'eager_OP_avg_time', 'compile_OP_avg_time', 'OP_speed_up_rate'
    ]
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns + [col for col in df.columns if col not in existing_columns]]
    
    # 保存到Excel
    df.to_excel(output_file, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description='提取日志文件中的训练时间信息和profile数据')
    parser.add_argument('--log_file', required=True, help='日志文件路径')
    parser.add_argument('--profile_dir', default='./profile', help='profile数据目录路径，默认为./profile')
    parser.add_argument('--output_file', default='analysis.xlsx', help='输出Excel文件路径，默认为log_analysis.xlsx')
    
    args = parser.parse_args()
    
    result = extract_log_info(
        log_file=args.log_file,
        profile_dir=args.profile_dir,
        output_file=args.output_file
    )
    print(f"提取完成，共找到{len(result)}个模型")
    print("\n提取结果:")
    print(result.head())

# 使用示例
if __name__ == "__main__":
    main()