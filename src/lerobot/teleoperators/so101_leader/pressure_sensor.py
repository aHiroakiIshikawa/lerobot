#!/usr/bin/env python

import logging
import re
import serial
import time
from threading import Lock, Thread

logger = logging.getLogger(__name__)


class PressureSensor:
    """Arduino Nano R4から圧力センサの値を読み取るクラス"""

    def __init__(self, port: str = "/dev/cu.usbmodem21101", baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection: serial.Serial | None = None
        self._latest_force_n = 0.0
        self._lock = Lock()
        self._running = False
        self._thread: Thread | None = None

    def connect(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        """
        シリアル接続を開始（リトライ機能付き）
        
        Args:
            max_retries: 最大リトライ回数
            retry_delay: リトライ間隔（秒）
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to pressure sensor on {self.port} (attempt {attempt + 1}/{max_retries})")
                
                self.serial_connection = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout
                )
                time.sleep(2)  # Arduinoのリセット待ち
                logger.info(f"Pressure sensor connected on {self.port}")
                
                # バックグラウンドで値を読み取るスレッドを開始
                self._running = True
                self._thread = Thread(target=self._read_loop, daemon=True)
                self._thread.start()
                
                return  # 成功
                
            except serial.SerialException as e:
                last_error = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        # 全リトライ失敗
        raise serial.SerialException(f"Failed to connect after {max_retries} attempts: {last_error}")

    def disconnect(self) -> None:
        """シリアル接続を終了"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            logger.info("Pressure sensor disconnected")

    def _read_loop(self) -> None:
        """バックグラウンドで継続的に圧力値を読み取る"""
        while self._running and self.serial_connection:
            try:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    
                    # "F = 0.123 [N]" のようなフォーマットから数値を抽出
                    match = re.search(r'F = ([\d.]+) \[N\]', line)
                    if match:
                        force_n = float(match.group(1))
                        with self._lock:
                            self._latest_force_n = force_n
                            
            except Exception as e:
                if self._running:
                    logger.warning(f"Error reading from pressure sensor: {e}")
                time.sleep(0.005)

    def get_force(self) -> float:
        """
        最新の圧力値を取得 [N]
        
        Returns:
            float: 圧力値 [N]
        """
        with self._lock:
            return self._latest_force_n

    @property
    def is_connected(self) -> bool:
        return self.serial_connection is not None and self.serial_connection.is_open