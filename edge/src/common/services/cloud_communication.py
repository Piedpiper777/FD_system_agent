"""
云边通信服务 - 从原services/cloud目录移过来
"""

import os
import json
import requests
import time
from queue import Queue
import yaml
import sys
from pathlib import Path

# 通过包结构自动导入模块
from config import Config


class CloudCommunicationService:
    """云边通信服务"""

    def __init__(self, communication_mode='auto'):
        """
        初始化通信服务
        communication_mode: 'auto', 'http', 'grpc', 'queue'
        """
        self.mode = communication_mode
        
        # 延迟获取配置，避免在应用上下文外访问 current_app
        self.cloud_base_url = None

        self.request_queue = Queue()
        self.response_queue = Queue()

        # 从配置文件加载通信参数
        self._load_communication_config()

        # 根据模式初始化
        if self.mode == 'auto':
            self._init_auto_mode()
        elif self.mode == 'http':
            self._init_http_client()
        elif self.mode == 'queue':
            self._init_message_queue()
        elif self.mode == 'grpc':
            self._init_grpc_client()
    
    def _get_cloud_base_url(self):
        """获取云端服务URL（延迟获取，确保在应用上下文中）"""
        if self.cloud_base_url is None:
            try:
                from flask import current_app
                self.cloud_base_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
            except RuntimeError:
                # 不在 Flask 应用上下文中，使用默认值
                self.cloud_base_url = 'http://localhost:5001'
        return self.cloud_base_url

    def _load_communication_config(self):
        """加载通信配置"""
        try:
            # 获取配置文件路径
            base_path = Path(__file__).resolve().parents[4]  # 项目根目录
            config_path = base_path / 'config' / 'default.yaml'
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                comm_config = config.get('communication', {})
                self.max_retries = comm_config.get('max_retries', 3)
                self.retry_delay = comm_config.get('retry_delay', 1.0)
                self.request_timeout = comm_config.get('request_timeout', 30)
                self.long_request_timeout = comm_config.get('long_request_timeout', 300)
                self.training_timeout = comm_config.get('training_timeout', 3600)
            else:
                self._use_default_config()
        except Exception as e:
            print(f"Failed to load communication config: {e}")
            self._use_default_config()

    def _use_default_config(self):
        """使用默认配置"""
        self.max_retries = 3
        self.retry_delay = 1.0
        self.request_timeout = 30
        self.long_request_timeout = 300
        self.training_timeout = 3600

    def _init_auto_mode(self):
        """自动模式初始化：只使用HTTP云端通信"""
        try:
            self._init_http_client()
            if self.mode == 'http':
                print("Auto mode: Using HTTP cloud communication")
            else:
                print("Auto mode: Cloud service unavailable")
        except:
            print("Auto mode: HTTP initialization failed")
            self.mode = 'http'

    def _init_http_client(self):
        """初始化HTTP客户端"""
        try:
            response = requests.get(f"{self._get_cloud_base_url()}/api/health/", timeout=5)
            if response.status_code == 200:
                print("Cloud HTTP service available")
            else:
                print("Cloud HTTP service not available")
                self.mode = 'http'
        except:
            print("Cloud HTTP service not available")
            self.mode = 'http'

    def _init_grpc_client(self):
        """初始化gRPC客户端"""
        raise NotImplementedError('gRPC mode is not yet implemented.')

    def _init_message_queue(self):
        """初始化消息队列"""
        raise NotImplementedError('Message queue mode is not yet implemented.')

    def call_cloud_inference(self, request_data):
        """调用云端推理服务"""
        if self.mode == 'http':
            return self._call_http_inference(request_data)
        elif self.mode == 'grpc':
            return self._call_grpc_inference(request_data)
        elif self.mode == 'queue':
            return self._call_queue_inference(request_data)
        else:
            return self._call_subprocess_inference(request_data)

    def call_cloud_training(self, request_data):
        """调用云端训练服务"""
        if self.mode == 'http':
            return self._call_http_training(request_data)
        elif self.mode == 'grpc':
            return self._call_grpc_training(request_data)
        elif self.mode == 'queue':
            return self._call_queue_training(request_data)
        else:
            return self._call_subprocess_training(request_data)

    def _call_http_inference(self, request_data):
        """通过HTTP调用云端推理"""
        try:
            response = self._make_request_with_retry(
                'POST',
                f"{self._get_cloud_base_url()}/inference",
                json=request_data,
                timeout=self.long_request_timeout
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                return response.json()
            elif response.status_code >= 500:
                return {
                    'success': False,
                    'error': f'Server error after retries: HTTP {response.status_code}: {response.text}'
                }
            else:
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}: {response.text}'
                }
        except requests.exceptions.Timeout:
            return {'success': False, 'error': f'Inference timeout after {self.max_retries} retries'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': f'Connection failed after {self.max_retries} retries'}
        except Exception as e:
            return {'success': False, 'error': f'Inference request failed: {str(e)}'}

    def _call_http_training(self, request_data):
        """通过HTTP调用云端训练"""
        try:
            response = self._make_request_with_retry(
                'POST',
                f"{self._get_cloud_base_url()}/training",
                json=request_data,
                timeout=self.training_timeout
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                return response.json()
            elif response.status_code >= 500:
                return {
                    'success': False,
                    'error': f'Server error after retries: HTTP {response.status_code}: {response.text}'
                }
            else:
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}: {response.text}'
                }
        except requests.exceptions.Timeout:
            return {'success': False, 'error': f'Training timeout after {self.max_retries} retries'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': f'Connection failed after {self.max_retries} retries'}
        except Exception as e:
            return {'success': False, 'error': f'Training request failed: {str(e)}'}

    def _call_subprocess_inference(self, request_data):
        """通过subprocess调用云端推理（已移除，不再支持）"""
        return {
            'success': False,
            'error': 'Subprocess mode is no longer supported. Cloud services must be deployed remotely.'
        }

    def _call_subprocess_training(self, request_data):
        """通过subprocess调用云端训练"""
        raise NotImplementedError('Subprocess mode is no longer supported. Cloud services must be deployed remotely.')

    def _call_grpc_inference(self, request_data):
        """通过gRPC调用云端推理"""
        raise NotImplementedError('gRPC mode is not yet implemented.')

    def _call_grpc_training(self, request_data):
        """通过gRPC调用云端训练"""
        raise NotImplementedError('gRPC mode is not yet implemented.')

    def _call_queue_inference(self, request_data):
        """通过消息队列调用云端推理"""
        raise NotImplementedError('Message queue mode is not yet implemented.')

    def _call_queue_training(self, request_data):
        """通过消息队列调用云端训练"""
        raise NotImplementedError('Message queue mode is not yet implemented.')

    def get_communication_status(self):
        """获取通信状态"""
        return {
            'mode': self.mode,
            'cloud_service_available': self._check_cloud_availability(),
            'queue_size': self.request_queue.qsize()
        }

    def _check_cloud_availability(self):
        """检查云端服务可用性"""
        if self.mode == 'http':
            try:
                response = requests.get(f"{self._get_cloud_base_url()}/api/health/", timeout=5)
                return response.status_code == 200
            except:
                return False
        else:
            return False

    def _make_request_with_retry(self, method, url, **kwargs):
        """带重试机制的HTTP请求"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = requests.request(method, url, **kwargs)

                if response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue

                return response

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                break

        if last_exception:
            raise last_exception
        else:
            raise requests.exceptions.RequestException("Max retries exceeded")
