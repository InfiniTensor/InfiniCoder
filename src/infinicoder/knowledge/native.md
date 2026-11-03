# 原生代码开发工作法（教学材料，不是验收证据）

先确认 ABI、类型/形状、所有权、构建参数与工具链，再实现一个可编译切片。
从实际头文件和调用点读取接口；不要根据相近框架补造成员。
编译失败先修首个根因，再重新编译，避免同时修改大量级联诊断。
C++ 命名函数和 CUDA kernel 必须定义在命名空间或文件作用域，不能嵌套在函数内。
区分 host/device 指针、元素数与字节数、尺寸与 stride、同步与异步生命周期。
使用哈希守卫的小块替换；多文件变更按接口声明、实现、调用点分成验收单元。
compile_commands 只说明构建身份；正则声明检索不是类型解析或真实 call graph。
输出被截断时不能执行片段；缩小工作单元或改用聚焦行范围编辑。

## General C++ declaration and exception rules

- `noexcept` does not catch or suppress exceptions. An exception escaping such a function terminates the process. If a contract requires cleanup to suppress exceptions, its implementation must enforce that behavior.
- Moving an exclusive resource transfers its ownership obligation. The moved-from object must remain valid and cannot retain that same release obligation. Self-assignment needs its own ownership analysis.
- Include the declaration provider directly: `<cmath>` for mathematical functions, `<algorithm>` for range algorithms, `<cstddef>` for `std::size_t`, `<utility>` for move utilities. A neighboring header need not include these transitively. Use compiler evidence to confirm missing declarations; adding headers cannot fix wrong mathematics.
