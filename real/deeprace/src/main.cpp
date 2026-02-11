#include "../include/ICM42688/src/ICM42688.h"
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

static constexpr uint8_t IMU_ADDR = 0x68;
static constexpr int SDA_PIN = 21;
static constexpr int SCL_PIN = 22;
ICM42688 IMU(Wire, IMU_ADDR, SDA_PIN, SCL_PIN);

const char *WIFI_SSID = "ssid";
const char *WIFI_PASS = "pass";
// const char *WIFI_SSID = "ssid";
// const char *WIFI_PASS = "pass";

static constexpr uint16_t IMU_PORT_TX = 9000; // ESP32 -> PC
static constexpr uint16_t CMD_PORT_RX = 9001; // PC -> ESP32

WiFiUDP udp;
IPAddress bcast;

static constexpr int LF_IN1 = 25, LF_IN2 = 26;
static constexpr int LR_IN1 = 27, LR_IN2 = 32;

static constexpr int RF_IN1 = 18, RF_IN2 = 19;
static constexpr int RR_IN1 = 16, RR_IN2 = 17;

// PWM config
static constexpr int PWM_FREQ = 20000;
static constexpr int PWM_RES = 10;
static constexpr int PWM_MAX = (1 << PWM_RES) - 1;

static constexpr int CH_LF1 = 0, CH_LF2 = 1;
static constexpr int CH_LR1 = 2, CH_LR2 = 3;
static constexpr int CH_RF1 = 4, CH_RF2 = 5;
static constexpr int CH_RR1 = 6, CH_RR2 = 7;

static float left_cmd = 0.0f;
static float right_cmd = 0.0f;

void wifiInit();
void imuInit();
void motorsInit();

void setMotor(float s, int in1, int in2, int ch1, int ch2);
void applyDrive(float left, float right);

bool readCommandUDP(float &left, float &right);

void setup() {
  Serial.begin(115200);
  delay(300);

  wifiInit();
  imuInit();
  motorsInit();

  Serial.println("Ready: IMU->UDP(9000 bcast), CMD<-UDP(9001)");
}

void loop() {
  float l, r;
  if (readCommandUDP(l, r)) {
    left_cmd = l;
    right_cmd = r;
    applyDrive(left_cmd, right_cmd);
  }

  int st = IMU.getAGT();
  if (st >= 0 && WiFi.status() == WL_CONNECTED) {
    float ax = IMU.accX(), ay = IMU.accY(), az = IMU.accZ();
    float gx = IMU.gyrX(), gy = IMU.gyrY(), gz = IMU.gyrZ();
    float tC = IMU.temp();

    char msg[128];
    int n = snprintf(msg, sizeof(msg), "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f\n",
                     ax, ay, az, gx, gy, gz, tC);

    udp.beginPacket(bcast, IMU_PORT_TX);
    udp.write((const uint8_t *)msg, n);
    udp.endPacket();
  }

  delay(20);
}

void wifiInit() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  bcast = WiFi.localIP();
  bcast[3] = 255;

  udp.begin(CMD_PORT_RX);
}

void imuInit() {
  Wire.begin(SDA_PIN, SCL_PIN);
  int status = IMU.begin();
  if (status < 0) {
    Serial.println("ICM42688 init failed.");
    while (true)
      delay(1000);
  }
  IMU.setAccelFS(ICM42688::gpm4);
  IMU.setGyroFS(ICM42688::dps500);
  Serial.println("ICM-42688-P ready (UDP streaming).");
}

#include "esp_arduino_version.h"

static void pwmAttach(int pin, int ch) {
#if defined(ESP_ARDUINO_VERSION) &&                                            \
    (ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0))
  bool ok = ledcAttachChannel(pin, PWM_FREQ, PWM_RES, ch);
  Serial.printf("LEDC attach pin=%d ch=%d ok=%d\n", pin, ch, (int)ok);
  ledcWriteChannel(ch, 0);
#else
  uint32_t f = ledcSetup(ch, PWM_FREQ, PWM_RES);
  ledcAttachPin(pin, ch);
  Serial.printf("LEDC setup ch=%d pin=%d f=%lu\n", ch, pin, (unsigned long)f);
  ledcWrite(ch, 0);
#endif
}

static void pwmWriteCh(int ch, int duty) {
#if defined(ESP_ARDUINO_VERSION) &&                                            \
    (ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0))
  ledcWriteChannel(ch, duty);
#else
  ledcWrite(ch, duty);
#endif
}

static void attachPWM(int pin, int ch) {
#if defined(ESP_ARDUINO_VERSION) &&                                            \
    (ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0))
  bool ok = ledcAttachChannel(pin, PWM_FREQ, PWM_RES, ch);
  Serial.printf("LEDC attach pin=%d ch=%d ok=%d\n", pin, ch, (int)ok);
  ledcWriteChannel(ch, 0);
#else
  uint32_t f = ledcSetup(ch, PWM_FREQ, PWM_RES);
  ledcAttachPin(pin, ch);
  Serial.printf("LEDC setup ch=%d pin=%d f=%lu\n", ch, pin, (unsigned long)f);
  ledcWrite(ch, 0);
#endif
}

void motorsInit() {
  attachPWM(LF_IN1, CH_LF1);
  attachPWM(LF_IN2, CH_LF2);
  attachPWM(LR_IN1, CH_LR1);
  attachPWM(LR_IN2, CH_LR2);
  attachPWM(RF_IN1, CH_RF1);
  attachPWM(RF_IN2, CH_RF2);
  attachPWM(RR_IN1, CH_RR1);
  attachPWM(RR_IN2, CH_RR2);

  applyDrive(0, 0);
  Serial.println("Motors initialized.");
}

void setMotor(float s, int in1, int in2, int ch1, int ch2) {
  if (s > 1.0f)
    s = 1.0f;
  if (s < -1.0f)
    s = -1.0f;
  s = in1 == LF_IN1 || in1 == LR_IN1 ? -s : s;

  int duty = (int)(fabsf(s) * PWM_MAX);

  Serial.printf("Motor duty=%d, fwd=%d\n", duty, s > 0);

  if (duty == 0) {
    pwmWriteCh(ch1, 0);
    pwmWriteCh(ch2, 0);
    return;
  }

  if (s > 0) {
    pwmWriteCh(ch1, duty);
    pwmWriteCh(ch2, 0);
  } else {
    pwmWriteCh(ch1, 0);
    pwmWriteCh(ch2, duty);
  }
}

void applyDrive(float left, float right) {
  setMotor(left, LF_IN1, LF_IN2, CH_LF1, CH_LF2);
  setMotor(left, LR_IN1, LR_IN2, CH_LR1, CH_LR2);

  setMotor(right, RF_IN1, RF_IN2, CH_RF1, CH_RF2);
  setMotor(right, RR_IN1, RR_IN2, CH_RR1, CH_RR2);
}

bool readCommandUDP(float &left, float &right) {
  int packetSize = udp.parsePacket();
  if (packetSize <= 0)
    return false;

  char buf[64];
  int len = udp.read(buf, sizeof(buf) - 1);
  if (len <= 0)
    return false;
  buf[len] = '\0';

  float l, r;
  if (sscanf(buf, "%f,%f", &l, &r) == 2) {
    Serial.printf("CMD left=%.3f right=%.3f\n", l, r);
    left = l;
    right = r;
    return true;
  }
  return false;
}

// #include <Arduino.h>

// static constexpr int AIN1 = 16; // change to the pins you wired
// static constexpr int AIN2 = 17;

// void setup() {
//   pinMode(AIN1, OUTPUT);
//   pinMode(AIN2, OUTPUT);

//   // full forward for 2 seconds

// }

// void loop() {

//   digitalWrite(AIN1, LOW);
//   digitalWrite(AIN2, HIGH);
//   delay(2000);

//   // stop
//   digitalWrite(AIN1, LOW);
//   digitalWrite(AIN2, LOW);
//   delay(2000);
// }
