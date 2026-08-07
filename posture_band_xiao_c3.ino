#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

// XIAO ESP32-C3 D1 is GPIO3.
// Connect the YwRobot vibration module signal pin to D1.
constexpr int MOTOR_PIN = D1;

constexpr char DEVICE_NAME[] = "PostureBand";
constexpr char SERVICE_UUID[] =
    "7d2a4b20-8f77-4e24-9a63-94a4ef0d12b2";
constexpr char CHARACTERISTIC_UUID[] =
    "7d2a4b21-8f77-4e24-9a63-94a4ef0d12b2";

constexpr unsigned long WARNING_STEP_MS = 150;  // 0.15 seconds
constexpr int WARNING_PULSE_COUNT = 2;
constexpr unsigned long COMMAND_TIMEOUT_MS = 5000;

enum class PostureState {
  OFF,
  NORMAL,
  WARNING,
  BAD,
  NO_POSE
};

volatile bool deviceConnected = false;
PostureState currentState = PostureState::OFF;
unsigned long lastCommandMs = 0;
unsigned long warningStepStartedMs = 0;
int warningStep = 0;
bool warningRunning = false;

void motorOn() {
  digitalWrite(MOTOR_PIN, HIGH);
}

void motorOff() {
  digitalWrite(MOTOR_PIN, LOW);
}

void startWarningPattern() {
  warningStep = 0;
  warningStepStartedMs = millis();
  warningRunning = true;
  motorOn();
}

void applyState(PostureState newState) {
  // Heartbeat packets repeat the same text. Do not restart the two-pulse
  // WARNING pattern unless the posture state actually changed.
  if (newState == currentState) {
    return;
  }

  currentState = newState;
  warningRunning = false;

  if (currentState == PostureState::WARNING) {
    startWarningPattern();
  } else if (currentState == PostureState::BAD) {
    motorOn();
  } else {
    motorOff();
  }
}

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override {
    deviceConnected = true;
    lastCommandMs = millis();
  }

  void onDisconnect(BLEServer* server) override {
    deviceConnected = false;
    currentState = PostureState::OFF;
    warningRunning = false;
    motorOff();
    BLEDevice::startAdvertising();
  }
};

class StateCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* characteristic) override {
    String command = characteristic->getValue().c_str();
    command.trim();
    command.toUpperCase();
    lastCommandMs = millis();

    if (command == "WARNING") {
      applyState(PostureState::WARNING);
    } else if (command == "BAD") {
      applyState(PostureState::BAD);
    } else if (command == "NORMAL") {
      applyState(PostureState::NORMAL);
    } else if (command == "NO_POSE") {
      applyState(PostureState::NO_POSE);
    } else {
      applyState(PostureState::OFF);
    }
  }
};

void setup() {
  pinMode(MOTOR_PIN, OUTPUT);
  motorOff();

  Serial.begin(115200);
  BLEDevice::init(DEVICE_NAME);

  BLEServer* server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService* service = server->createService(SERVICE_UUID);
  BLECharacteristic* characteristic = service->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_WRITE);
  characteristic->setCallbacks(new StateCallbacks());

  service->start();

  BLEAdvertising* advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  advertising->start();

  Serial.println("PostureBand BLE ready");
}

void loop() {
  const unsigned long now = millis();

  if (warningRunning && now - warningStepStartedMs >= WARNING_STEP_MS) {
    warningStepStartedMs = now;
    warningStep++;

    if (warningStep >= WARNING_PULSE_COUNT * 2) {
      warningRunning = false;
      motorOff();
    } else if (warningStep % 2 == 0) {
      motorOn();
    } else {
      motorOff();
    }
  }

  // Fail safe: stop vibration if the Raspberry Pi disappears or commands
  // stop arriving. The Pi sends a heartbeat every two seconds.
  if (deviceConnected && now - lastCommandMs > COMMAND_TIMEOUT_MS) {
    applyState(PostureState::OFF);
  }

  delay(5);
}
