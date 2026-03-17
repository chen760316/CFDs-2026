def convert_to_constant_cfds(dataset, cfd_list):
    constant_cfds = []

    # 遍历每个CFD
    for cfd in cfd_list:
        lhs_constants = {}
        lhs_variables = []
        rhs_attribute = None

        # 分离左属性集中的常量和变量属性
        for attr, value in cfd['left'].items():
            if value == '_':  # 变量属性
                lhs_variables.append(attr)
            else:  # 常量属性
                lhs_constants[attr] = value

        rhs_attribute = cfd['right']

        # 找到属性为常量的元组
        constant_tuples = []
        for record in dataset:
            is_match = True
            for attr, value in lhs_constants.items():
                if record[attr] != value:
                    is_match = False
                    break
            if is_match:
                constant_tuples.append(record)

        # 为变量属性生成对应的常量值
        for record in constant_tuples:
            for variable_combination in generate_variable_combinations(lhs_variables, dataset):
                constant_cfd = {'left': {}, 'right': rhs_attribute}
                constant_cfd['left'].update(lhs_constants)
                constant_cfd['left'].update(variable_combination)
                constant_cfds.append(constant_cfd)

    return constant_cfds

# 生成变量属性的所有可能取值组合
def generate_variable_combinations(variables, dataset):
    variable_combinations = []

    # 获取每个变量属性可能的取值集合
    variable_value_sets = {}
    for variable in variables:
        values = set(record[variable] for record in dataset)
        variable_value_sets[variable] = list(values)

    # 生成所有可能的组合
    from itertools import product
    for combination in product(*variable_value_sets.values()):
        variable_combinations.append({variables[i]: combination[i] for i in range(len(variables))})

    return variable_combinations

# 测试数据集
dataset = [
    {'relationship': 'Husband', 'capital-loss': 1000, 'sex': 'Male'},
    {'relationship': 'Husband', 'capital-loss': 2000, 'sex': 'Female'},
    {'relationship': 'Husband', 'capital-loss': 2000, 'sex': 'Male'},
]

# 原始CFDs列表
original_cfd_list = [
    {'left': {'capital-loss': '_', 'relationship': 'Husband'}, 'right': 'sex'}
]

# 转换为常量CFDs
constant_cfds = convert_to_constant_cfds(dataset, original_cfd_list)

# 输出结果
for cfd in constant_cfds:
    print(cfd)