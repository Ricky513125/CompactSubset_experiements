# 邮件通知设置指南

## 概述

训练脚本支持通过 Gmail 发送训练进度邮件通知。需要设置 `SENDER_PASSWORD` 环境变量来启用此功能。

## 设置步骤

### 1. 获取 Gmail 应用密码（App Password）

1. **登录 Google 账号**
   - 访问：https://myaccount.google.com/
   - 进入"安全性" → "两步验证"
   - 如果未启用两步验证，需要先启用

2. **生成应用专用密码**
   - 在"两步验证"页面底部，找到"应用专用密码"
   - 点击"选择应用" → 选择"邮件"
   - 点击"选择设备" → 选择"其他（自定义名称）"
   - 输入名称（如："训练脚本"或"Training Script"）
   - 点击"生成"
   - **复制生成的 16 位密码**（格式如：`abcd efgh ijkl mnop`）

### 2. 设置环境变量

#### 方法 1：临时设置（仅当前终端会话有效）

```bash
export SENDER_PASSWORD="your_16_digit_app_password"
```

#### 方法 2：永久设置（推荐）

将以下内容添加到 `~/.bashrc` 或 `~/.bash_profile`：

```bash
# Gmail 应用密码（用于训练脚本邮件通知）
export SENDER_PASSWORD="your_16_digit_app_password"
```

然后重新加载配置：

```bash
source ~/.bashrc
# 或
source ~/.bash_profile
```

#### 方法 3：在运行脚本时直接设置

```bash
SENDER_PASSWORD="your_16_digit_app_password" bash train_all_8B_sequential_no_C_no_inf_2_5.sh
```

### 3. 验证设置

运行以下命令检查是否设置成功：

```bash
if [ -z "$SENDER_PASSWORD" ]; then
    echo "❌ SENDER_PASSWORD 未设置"
else
    echo "✓ SENDER_PASSWORD 已设置（长度: ${#SENDER_PASSWORD}）"
fi
```

## 配置说明

### 默认配置

- **发送邮箱**: `lingyuli513125@gmail.com`（可通过 `SENDER_EMAIL` 环境变量修改）
- **收件邮箱**: `lilingyu513125@163.com`（可通过 `RECIPIENT_EMAIL` 环境变量修改）

### 自定义配置

如果需要修改发送邮箱或收件邮箱，可以设置环境变量：

```bash
export SENDER_EMAIL="your_email@gmail.com"
export RECIPIENT_EMAIL="recipient@example.com"
export SENDER_PASSWORD="your_app_password"
```

或者在脚本中直接修改：

```bash
# 编辑脚本文件
vim train_all_8B_sequential_no_C_no_inf_2_5.sh

# 修改以下行：
SENDER_EMAIL="${SENDER_EMAIL:-your_email@gmail.com}"
RECIPIENT_EMAIL="${RECIPIENT_EMAIL:-recipient@example.com}"
```

## 安全建议

1. **不要将密码提交到 Git**
   - 确保 `.bashrc` 或包含密码的文件不在 Git 仓库中
   - 或者使用环境变量文件（`.env`），并将其添加到 `.gitignore`

2. **使用应用专用密码**
   - 永远不要使用 Gmail 账号的普通密码
   - 只使用 Google 生成的应用专用密码

3. **定期更换密码**
   - 如果怀疑密码泄露，可以在 Google 账号设置中撤销并重新生成

## 故障排除

### 问题 1：邮件发送失败

**错误信息**: `⚠ 邮件发送失败: ...`

**可能原因**:
- 应用密码错误
- 两步验证未启用
- Gmail SMTP 被阻止

**解决方法**:
1. 检查应用密码是否正确（16位，无多余空格）
2. 确认已启用两步验证
3. 检查 Gmail 账号是否允许"不够安全的应用"访问（通常不需要，因为使用的是应用密码）

### 问题 2：未设置密码时脚本仍尝试发送邮件

**现象**: 脚本显示"未设置 SENDER_PASSWORD"，但继续运行

**说明**: 这是正常行为。如果未设置 `SENDER_PASSWORD`，脚本会跳过邮件发送，不影响训练任务。

## 测试邮件发送

可以创建一个简单的测试脚本来验证邮件配置：

```bash
#!/bin/bash
# test_email.sh

SENDER_EMAIL="${SENDER_EMAIL:-lingyuli513125@gmail.com}"
SENDER_PASSWORD="${SENDER_PASSWORD:-}"
RECIPIENT_EMAIL="${RECIPIENT_EMAIL:-lilingyu513125@163.com}"

if [ -z "$SENDER_PASSWORD" ]; then
    echo "错误: 未设置 SENDER_PASSWORD"
    exit 1
fi

python3 << EOF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    msg = MIMEMultipart()
    msg['From'] = "$SENDER_EMAIL"
    msg['To'] = "$RECIPIENT_EMAIL"
    msg['Subject'] = "测试邮件 - $(date '+%Y-%m-%d %H:%M:%S')"
    
    body = "这是一封测试邮件，用于验证 Gmail 应用密码配置。"
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("$SENDER_EMAIL", "$SENDER_PASSWORD")
    
    text = msg.as_string()
    server.sendmail("$SENDER_EMAIL", "$RECIPIENT_EMAIL", text)
    server.quit()
    
    print("✓ 测试邮件发送成功！")
except Exception as e:
    print(f"✗ 邮件发送失败: {e}")
    exit(1)
EOF
```

运行测试：

```bash
chmod +x test_email.sh
./test_email.sh
```

## 相关文件

- `train_all_8B_sequential_no_C_no_inf_2_5.sh`
- `train_all_8B_sequential_no_C_no_inf_6_8.sh`
- `train_all_Gemma3_27B_1_4.sh`
- 其他训练脚本

## 参考链接

- [Google 账号安全性设置](https://myaccount.google.com/security)
- [Gmail 应用专用密码帮助](https://support.google.com/accounts/answer/185833)
