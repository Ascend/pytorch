# Docker 镜像构建流程问题检查与修复报告

## 发现的问题

### 问题 1：定时构建和 push 触发时不推送镜像 ❌❌❌

**严重级别**：P0 - Critical

**问题描述**：

原 workflow 使用条件表达式：
```yaml
${{ inputs.push_image && '--push' || '' }}
```

当触发方式为 `push` 或 `schedule` 时：
- `inputs.push_image` 为空（undefined）
- 条件表达式返回空字符串
- **结果：定时构建和 push 触发时不推送镜像到远端！**

**影响**：
- 定时构建的镜像无法自动推送到 registry
- 需要手动触发才能推送镜像

---

### 问题 2：镜像推送时机设计不合理 ⚠️

**原设计**：
- 手动触发：根据 inputs.push_image 决定是否推送
- push/schedule 触发：不推送 ❌

**PyTorch 上游参考**：
```yaml
- name: Push to ghcr.io
  if: ${{ github.event_name == 'push' }}
```

PyTorch 只在 push 到 main 分支时推送镜像。

**正确逻辑**：
- workflow_dispatch：根据 inputs.push_image 决定 ✅
- push（分支推送）：默认推送 ✅
- schedule（定时构建）：默认推送 ✅
- pull_request：不推送 ✅

---

### 问题 3：Summary 中版本提取逻辑错误 ⚠️

**原代码**：
```bash
CANN_MAJOR=$(echo "${{ inputs.cann_version || env.CANN_STABLE }}" | sed 's/-beta.*//' | sed 's/-rc.*//' | sed 's/\.[0-9]*$//')
```

**问题**：
对于简化版本 `9.0`：
- `sed 's/\.[0-9]*$//'` 删除最后的 `.数字`
- `9.0` → 删除 `.0` → 结果是 `9` ❌
- 应该保持 `9.0`

**正确逻辑**：
```bash
if [[ "$CANN_INPUT" =~ ^[0-9]+\.[0-9]+$ ]]; then
  CANN_MAJOR="$CANN_INPUT"  # 简化版本直接使用
else
  CANN_MAJOR=$(echo "$CANN_INPUT" | grep -oP '^[0-9]+\.[0-9]+')  # 提取前两位
fi
```

---

### 问题 4：双重登录导致冗余 ⚠️

**原流程**：
```
Workflow: docker/login-action 登录
Script: login_registry 再次登录
```

虽然不会出错，但浪费时间，且可能导致登录状态混乱。

---

### 问题 5：登录时机不优化 ⚠️

**原执行顺序**：
```
parse_args → check_dependencies → parse_cann_version → login_registry → build_image
```

**问题**：
如果镜像已存在，build_image 会跳过构建，但 login_registry 已经执行了登录。

**优化**：
将登录移到确认需要构建之后：
```
parse_args → check_dependencies → parse_cann_version → build_image
                                           ↓
                                      检查镜像是否存在
                                           ↓
                                      确认需要构建 → login_registry → 构建
```

---

## 修复方案

### 修复 1：添加参数确定步骤 ✅

```yaml
- name: Determine build parameters
  id: params
  run: |
    # 确定是否推送镜像
    if [[ "${{ github.event_name }}" == "workflow_dispatch" ]]; then
      PUSH_IMAGE="${{ inputs.push_image }}"
    elif [[ "${{ github.event_name }}" == "push" || "${{ github.event_name }}" == "schedule" ]]; then
      PUSH_IMAGE="true"
    else
      PUSH_IMAGE="false"
    fi

    echo "push_image=${PUSH_IMAGE}" >> $GITHUB_OUTPUT
```

**效果**：
- workflow_dispatch：根据用户输入决定
- push/schedule：默认推送 ✅
- 其他情况：不推送

---

### 修复 2：正确配置 login-action ✅

```yaml
- name: Login to Quay.io
  if: ${{ steps.params.outputs.push_image == 'true' }}
  uses: docker/login-action@v3
  with:
    registry: ${{ env.REGISTRY }}
    username: ${{ secrets.QUAY_USERNAME }}
    password: ${{ secrets.QUAY_PASSWORD }}
```

**效果**：
- 只在需要推送时登录
- 使用官方推荐的 login-action
- 更安全地处理 secrets

---

### 修复 3：优化脚本登录逻辑 ✅

```bash
# 在 build_image 函数中
# 先检查镜像是否存在
if [[ "$FORCE_BUILD" == "false" && "$PUSH_IMAGE" == "true" ]]; then
    if docker pull "${REGISTRY}/${QUAY_ORG}/${IMAGE_NAME}:${first_tag}" &>/dev/null; then
        log_info "镜像已存在，跳过构建"
        return 0
    fi
fi

# 确认需要构建，执行登录
if [[ "$PUSH_IMAGE" == "true" ]]; then
    if [[ "${SKIP_DOCKER_LOGIN:-false}" != "true" ]]; then
        login_registry
    fi
fi
```

**效果**：
- 先检查镜像是否存在，避免不必要的登录
- 支持 SKIP_DOCKER_LOGIN 环境变量（用于 CI）
- 本地使用仍然会自动登录

---

### 修复 4：添加 SKIP_DOCKER_LOGIN 环境变量 ✅

**Workflow 中**：
```yaml
- name: Build and push image
  env:
    SKIP_DOCKER_LOGIN: true  # 已通过 login-action 登录
```

**效果**：
- 避免 workflow 中双重登录
- 脚本检测到此变量后跳过登录

---

### 修复 5：修正 Summary 版本提取 ✅

```bash
CANN_INPUT="${{ steps.params.outputs.cann_version }}"

if [[ "$CANN_INPUT" =~ ^[0-9]+\.[0-9]+$ ]]; then
  CANN_MAJOR="$CANN_INPUT"  # 简化版本直接使用
else
  CANN_MAJOR=$(echo "$CANN_INPUT" | grep -oP '^[0-9]+\.[0-9]+')
fi
```

**效果**：
- `9.0` → 保持为 `9.0` ✅
- `9.0.0-beta.2` → 提取为 `9.0` ✅
- `8.0` → 保持为 `8.0` ✅

---

## 修复后的完整流程

### Workflow 流程

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Determine build parameters                         │
│  ├─ 判断 event_name 类型                                     │
│  ├─ 确定 push_image (workflow_dispatch → inputs.push_image) │
│  │                    (push/schedule → true)                │
│  ├─ 确定 force_build                                         │
│  ├─ 确定 cann_version                                        │
│  └─ 输出到 GITHUB_OUTPUT                                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Setup Docker Buildx                                │
│  ├─ 配置 docker-container driver                            │
│  └─ 使用 moby/buildkit:latest                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Login to Quay.io (conditional)                     │
│  ├─ if: push_image == 'true'                                │
│  ├─ 使用 docker/login-action                                │
│  └─ 处理 secrets 安全                                        │
└─────────────────────────────────────────────────────────────┘
                           ↓ (如果需要推送)
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Build and push image                               │
│  ├─ 设置 SKIP_DOCKER_LOGIN=true                             │
│  ├─ 调用 build_image.sh                                     │
│  ├─ 根据 push_image 添加 --push 参数                        │
│  └─ 根据 force_build 添加 --force 参数                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Summary (always)                                   │
│  ├─ 生成构建报告                                             │
│  ├─ 显示镜像标签                                             │
│  └─ 显示 Python 切换方法                                     │
└─────────────────────────────────────────────────────────────┘
```

### 脚本内部流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. parse_args                                              │
│  ├─ 解析命令行参数                                           │
│  └─ 设置 REGISTRY、QUAY_ORG 等                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2. check_dependencies                                      │
│  ├─ 检查 docker 是否安装                                     │
│  └─ 检查 docker buildx                                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  3. parse_cann_version                                      │
│  ├─ 从版本映射表查找 URL                                     │
│  ├─ 提取完整版本和大版本                                     │
│  └─ 判断是否为 stable                                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  4. build_image                                             │
│  ├─ 生成镜像标签                                             │
│  ├─ 检查镜像是否已存在（如果 force=false && push=true）      │
│  │  └─ 如果存在 → return 0                                  │
│  ├─ 确认需要构建 → 登录（如果 SKIP_DOCKER_LOGIN != true）    │
│  ├─ 执行 docker buildx build                                │
│  │  ├─ --build-arg 传递 URL                                 │
│  │  ├─ --tag 添加多个标签                                    │
│  │  └─ --push 或 --load                                     │
│  └─ 输出构建信息                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 触发场景对比表

| 触发方式 | CANN 版本 | Push Image | Force Build | 登录 | 推送 |
|---------|-----------|------------|-------------|------|------|
| workflow_dispatch (默认) | 9.0 | true | false | ✅ | ✅ |
| workflow_dispatch (--push=false) | 9.0 | false | false | ❌ | ❌ |
| workflow_dispatch (--force) | 9.0 | true | true | ✅ | ✅ |
| push (dev_master) | 9.0 (stable) | true | false | ✅ | ✅ |
| schedule (周日) | 9.0 (stable) | true | false | ✅ | ✅ |
| pull_request | - | - | - | ❌ | ❌ |

---

## 验证清单

### ✅ 已验证

1. ✅ Workflow YAML 格式正确
2. ✅ Shell 脚本语法正确
3. ✅ 参数传递逻辑正确
4. ✅ 登录时机优化正确
5. ✅ SKIP_DOCKER_LOGIN 机制正确
6. ✅ 版本提取逻辑正确
7. ✅ 条件判断逻辑正确

### 🔄 需要在实际运行中验证

1. 🔄 定时构建是否正确推送镜像
2. 🔄 push 触发是否正确推送镜像
3. 🔄 镜像已存在时是否正确跳过构建
4. 🔄 Docker buildx --push 是否成功推送所有标签
5. 🔄 Quay.io 登录是否成功
6. 🔄 镜像标签是否正确生成和推送

---

## 后续建议

### 建议 1：添加构建失败通知

```yaml
- name: Notify on failure
  if: failure()
  run: |
    # 可以集成 Slack/Email 通知
    echo "::error::Build failed for CANN version ${{ steps.params.outputs.cann_version }}"
```

### 建议 2：添加镜像验证步骤

```yaml
- name: Verify pushed image
  if: ${{ steps.params.outputs.push_image == 'true' }}
  run: |
    docker pull "${{ env.REGISTRY }}/${{ env.QUAY_ORG }}/${{ env.IMAGE_NAME }}:cann${{ steps.params.outputs.cann_version }}"
    docker inspect --format='{{.Config.Env}}' "${{ env.REGISTRY }}/${{ env.QUAY_ORG }}/${{ env.IMAGE_NAME }}:cann${{ steps.params.outputs.cann_version }}"
```

### 建议 3：添加构建缓存

```yaml
- name: Build and push image
  uses: docker/build-push-action@v5
  with:
    context: .github/docker
    file: .github/docker/pytorch-npu-builder.Dockerfile
    push: ${{ steps.params.outputs.push_image == 'true' }}
    tags: |
      ${{ env.REGISTRY }}/${{ env.QUAY_ORG }}/${{ env.IMAGE_NAME }}:cann${{ steps.params.outputs.cann_version }}
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

---

## 修复文件列表

| 文件 | 修改内容 |
|------|---------|
| `.github/workflows/build-docker-image.yml` | ✅ 添加参数确定步骤<br>✅ 修正 login-action 条件<br>✅ 添加 SKIP_DOCKER_LOGIN<br>✅ 修正 Summary 版本提取 |
| `.github/scripts/build_image.sh` | ✅ 优化登录时机<br>✅ 添加 SKIP_DOCKER_LOGIN 支持<br>✅ 移除主函数中的 login_registry |

---

**检查时间**: 2026-05-06
**检查人**: Claude Code
**状态**: ✅ 所有问题已修复，等待实际运行验证