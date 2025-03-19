# 基础变量定义
PDM := pdm
DIST_DIR := dist
BUILD_DIR := build
SPEC_FILE := synclip.spec
PORT_MAIN := 6001
PID_DIR := pids
LOG_DIR := logs
TEMP_DIR := temp
TMUX_SESSION := synclip
TMUX_SOCKET := /root/tmux-server

# PDM 配置
PDM_FLAGS := --no-self -v

# 定义变量
SUDO := sudo
APT := apt-get
YUM := yum
APT_FLAGS := -y --no-install-recommends
YUM_FLAGS := -y
# 创建必要的目录结构
.PHONY: init
init:
	@echo "Initializing project structure..."
	@mkdir -p $(PID_DIR) $(LOG_DIR) $(TEMP_DIR)/audio $(TEMP_DIR)/images
	@touch $(LOG_DIR)/.gitkeep $(PID_DIR)/.gitkeep
	@echo "Project structure initialized"

# 确保目录存在
.PHONY: ensure-dirs
ensure-dirs:
	@mkdir -p $(PID_DIR) $(LOG_DIR) $(TEMP_DIR)/audio $(TEMP_DIR)/images


# 系统包管理器
# 检查是否有 sudo 权限
SUDO := $(shell which sudo)
ifeq ($(SUDO),)
    SUDO := 
else
    SUDO := sudo
endif

# 基础变量定义
PYTHON := python3
PIP := $(PYTHON) -m pip
PYPI_URL := https://pypi.tuna.tsinghua.edu.cn/simple

.PHONY: install-ffmpeg
install-ffmpeg:
	@echo "Cleaning all git configurations..."
	@git config --global --unset-all url."https://gitee.com/mirrors".insteadOf || true
	@git config --global --remove-section url."https://gitee.com/mirrors" || true
	@git config --global --unset-all url."https://gitee.com".insteadOf || true
	@git config --global --remove-section url."https://gitee.com" || true
	@echo "Current git config:"
	@git config --global --list || true
	
	@echo "Installing system FFmpeg..."
	@if command -v apt-get >/dev/null 2>&1; then \
		echo "Configuring apt mirrors..." && \
		$(SUDO) cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
		$(SUDO) sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
		$(SUDO) apt-get update && \
		$(SUDO) apt-get install -y \
			ffmpeg \
			libavcodec-extra \
			python3-pip \
			python3-dev \
			build-essential; \
	fi
	
	@echo "Upgrading pip..."
	@$(PIP) install --upgrade pip -i $(PYPI_URL)
	
	@echo "Installing Python dependencies..."
	@$(PIP) install -i $(PYPI_URL) ffmpeg-python
	@GIT_SSL_NO_VERIFY=true $(PIP) install --no-cache-dir git+https://github.com/openai/whisper.git
	
	@echo "✅ Installation completed"




.PHONY: check-fonts
check-fonts:

	@echo "Checking installed fonts..."
	@fc-list | grep -i noto || echo "Warning: Noto fonts not found"
	@fc-list | grep -i dejavu || echo "Warning: DejaVu fonts not found"
	@echo "Font cache information:"
	@fc-cache -v
	@echo "✅ Font check complete."



.PHONY: install
install: system-deps install-ffmpeg check-fonts
	@echo "Installing dependencies..."
	@if ! command -v pdm >/dev/null 2>&1; then \
		echo "PDM not found. Installing PDM..."; \
		curl -sSL https://pdm-project.org/install-pdm.py | python3 - ; \
	fi
	@echo "Configuring PDM to use mirrors..."
	@$(PDM) config pypi.url https://pypi.tuna.tsinghua.edu.cn/simple
	@if [ ! -d .venv ]; then \
		echo "Creating virtual environment..."; \
		$(PDM) venv create -f; \
	fi
	@echo "Installing project dependencies..."
	@$(PDM) install $(PDM_FLAGS) || \
	(echo "Dependencies installation failed, retrying in 3 seconds..." && \
	sleep 3 && \
	$(PDM) install $(PDM_FLAGS))
	@echo "✅ Installation completed successfully!"


.PHONY: reinstall
reinstall: clean system-deps
	@echo "Reinstalling all dependencies..."
	@rm -rf .venv pdm.lock
	@$(MAKE) install
	@echo "✅ Reinstallation completed successfully!"


# 系统依赖安装
.PHONY: system-deps
system-deps:
	@echo "Installing system dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		$(SUDO) $(APT) update && \
		$(SUDO) $(APT) install $(APT_FLAGS) \
			ffmpeg \
			python3-dev \
			python3-venv \
			binutils \
			libsox-dev \
			sox \
			libsox-fmt-all \
			libtbb12 \
			libtbb-dev \
			build-essential \
			libsm6 \
			libxext6 \
			libxrender-dev \
			libglib2.0-0 \
			libgl1-mesa-glx \
			fonts-noto-cjk \
			fonts-noto-color-emoji \
			fonts-dejavu \
			fonts-dejavu-core \
			fonts-dejavu-extra \
			fontconfig; \
	elif command -v yum >/dev/null 2>&1; then \
		$(SUDO) $(YUM) install $(YUM_FLAGS) \
			ffmpeg \
			python3-devel \
			python3-venv \
			binutils \
			sox-devel \
			sox \
			tbb-devel \
			gcc \
			gcc-c++ \
			libSM \
			libXext \
			libXrender \
			glib2 \
			mesa-libGL \
			google-noto-cjk-fonts \
			google-noto-emoji-fonts \
			dejavu-fonts-common \
			dejavu-sans-fonts \
			dejavu-serif-fonts \
			fontconfig; \
	else \
		echo "Warning: Could not detect package manager. Please install dependencies manually."; \
	fi
	@# 更新字体缓存
	@$(SUDO) fc-cache -fv
	@echo "✅ System dependencies installed successfully."



# 清理
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete."


# 查看日志
.PHONY: logs
logs:
	@echo "Checking tmux sessions and logs..."
	@if ! command -v tmux >/dev/null 2>&1; then \
		echo "tmux is not installed"; \
		exit 1; \
	fi
	
	@# 检查 tmux 会话
	@if tmux has-session -t $(TMUX_SESSION) 2>/dev/null; then \
		echo "Attaching to tmux session $(TMUX_SESSION)..."; \
		echo "To detach, press Ctrl+B then D"; \
		tmux attach-session -t $(TMUX_SESSION); \
	else \
		if [ -f "$(LOG_DIR)/app.log" ]; then \
			echo "No active session found. Showing last 50 lines of log file:"; \
			echo "----------------------------------------"; \
			tail -n 50 $(LOG_DIR)/app.log; \
			echo "----------------------------------------"; \
			echo "To see full log, use: cat $(LOG_DIR)/app.log"; \
		else \
			echo "No active sessions or log files found."; \
		fi \
	fi

# 停止所有服务
.PHONY: stop
stop:
	@echo "Stopping all services..."
	
	@# 关闭所有 tmux 会话
	@if tmux has-session -t $(TMUX_SESSION) 2>/dev/null; then \
		echo "Killing tmux session: $(TMUX_SESSION)"; \
		tmux kill-session -t $(TMUX_SESSION); \
	fi
	
	@# 关闭所有其他 tmux 会话
	@for session in $$(tmux ls 2>/dev/null | cut -d ':' -f 1); do \
		echo "Killing tmux session: $$session"; \
		tmux kill-session -t "$$session" 2>/dev/null || true; \
	done
	
	@# 清理文件
	@if [ -f $(PID_DIR)/app.pid ]; then \
		echo "Removing PID file"; \
		rm $(PID_DIR)/app.pid; \
	fi
	
	@if [ -f $(LOG_DIR)/app.log ]; then \
		echo "Archiving log file"; \
		mv $(LOG_DIR)/app.log $(LOG_DIR)/app.log.old 2>/dev/null || true; \
	fi
	
	@echo "✅ All services stopped and cleaned up successfully"



.PHONY: build_3090
build_3090: install
	@echo "Building executable..."
	@# 确保 pyinstaller 已安装
	@$(PDM) install -G dev --no-self $(PDM_FLAGS)
	@if [ ! -d "$(DIST_DIR)" ]; then \
		mkdir -p $(DIST_DIR); \
	fi
	@if [ -f "$(DIST_DIR)/seemingai-server" ]; then \
		echo "Backing up existing executable..."; \
		mv $(DIST_DIR)/seemingai-server $(DIST_DIR)/seemingai-server.backup; \
	fi
	@echo "Running PyInstaller build..."
	@$(PDM) run pyinstaller --clean \
		--onefile \
		--name seemingai-server \
		--distpath $(DIST_DIR) \
		--workpath $(BUILD_DIR) \
		--hidden-import=numba.core.types.old_scalars \
		--hidden-import=numba.core.types \
		--hidden-import=numba.core \
		--hidden-import=numba \
		--hidden-import=whisper \
		--collect-all numba \
		--collect-all whisper \
		app.py || \
	(echo "Build failed. Check logs for details"; exit 1)
	@if [ -f "$(DIST_DIR)/seemingai-server" ]; then \
		echo "✅ Build completed successfully."; \
		echo "Executable location: $(DIST_DIR)/seemingai-server"; \
		echo "File size: $$(du -h $(DIST_DIR)/seemingai-server | cut -f1)"; \
	else \
		echo "❌ Build failed: Executable not found"; \
		exit 1; \
	fi





.PHONY: build_4090
build_4090: install
	@echo "Building executable..."
	@# Install pyinstaller if not already installed
	@$(PDM) add --dev pyinstaller $(PDM_FLAGS) || (echo "Failed to install pyinstaller"; exit 1)
	@if [ ! -d "$(DIST_DIR)" ]; then \
		mkdir -p $(DIST_DIR); \
	fi
	@if [ -f "$(DIST_DIR)/seemingai-server" ]; then \
		echo "Backing up existing executable..."; \
		mv $(DIST_DIR)/seemingai-server $(DIST_DIR)/seemingai-server.backup; \
	fi
	@echo "Running PyInstaller build..."
	@$(PDM) run pyinstaller --clean \
		--onefile \
		--name seemingai-server \
		--distpath $(DIST_DIR) \
		--workpath $(BUILD_DIR) \
		--hidden-import=numba.core.types.old_scalars \
		--hidden-import=numba.core.types \
		--hidden-import=numba.core \
		--hidden-import=numba \
		--hidden-import=whisper \
		--collect-all numba \
		--collect-all whisper \
		app.py || \
	(echo "Build failed. Check logs for details"; exit 1)
	@if [ -f "$(DIST_DIR)/seemingai-server" ]; then \
		echo "✅ Build completed successfully."; \
		echo "Executable location: $(DIST_DIR)/seemingai-server"; \
		echo "File size: $$(du -h $(DIST_DIR)/seemingai-server | cut -f1)"; \
	else \
		echo "❌ Build failed: Executable not found"; \
		exit 1; \
	fi


.PHONY: check-env
check-env:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found in project root directory"; \
		exit 1; \
	fi

# 生产模式启动
.PHONY: start
start: ensure-dirs check-env
	@echo "Starting server in production mode..."
	@if [ -f $(PID_DIR)/app.pid ]; then \
		echo "Server is already running"; \
		exit 1; \
	fi
	
	@# 确保 tmux 已安装
	@if ! command -v tmux >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y tmux; \
	fi
	
	@# 确保没有同名会话存在
	@tmux kill-session -t $(TMUX_SESSION) 2>/dev/null || true
	
	@# 确保日志目录存在
	@mkdir -p $(LOG_DIR)
	
	@echo "Creating new tmux session..."
	@# 创建新的 tmux 会话并设置工作目录
	@tmux new-session -d -s $(TMUX_SESSION) -c $(PWD)
	@tmux send-keys -t $(TMUX_SESSION) \
		"cd $(PWD) && echo 'Starting server...' && PYTHONPATH=$(PWD) $(DIST_DIR)/seemingai-server --port $(PORT_MAIN) 2>&1 | tee $(LOG_DIR)/app.log" C-m
	
	@# 等待服务启动
	@sleep 2
	
	@# 检查服务是否成功启动
	@if tmux has-session -t $(TMUX_SESSION) 2>/dev/null; then \
		if pgrep -f "seemingai-server.*$(PORT_MAIN)" > /dev/null; then \
			echo "Server started successfully in tmux session: $(TMUX_SESSION)"; \
			pgrep -f "seemingai-server.*$(PORT_MAIN)" > $(PID_DIR)/app.pid; \
			echo "Production server started on port $(PORT_MAIN)"; \
			echo "To view logs in real-time, use: make logs"; \
			echo "To detach from logs view, press Ctrl+B then D"; \
		else \
			echo "Server process not found. Check logs for errors:"; \
			tail -n 20 $(LOG_DIR)/app.log; \
			$(MAKE) stop; \
			exit 1; \
		fi \
	else \
		echo "Failed to start server. Check logs for details."; \
		exit 1; \
	fi



.PHONY: restart
restart: stop
	@sleep 2
	@$(MAKE) start


	
.PHONY: commit
commit:
	@echo "Committing changes..."
	@# First check if git identity is configured
	@if [ -z "$$(git config user.name)" ] || [ -z "$$(git config user.email)" ]; then \
		echo "Git identity not configured. Please run:"; \
		echo "  git config --global user.email \"you@example.com\""; \
		echo "  git config --global user.name \"Your Name\""; \
		echo "Or without --global to set identity only in this repository."; \
		exit 1; \
	fi
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Changes detected, committing..."; \
		git add .; \
		git commit -m "update" --no-verify; \
		git push origin main; \
		echo "Changes pushed successfully"; \
	else \
		echo "No changes to commit"; \
	fi