#include <DHT.h>
#include <Adafruit_NeoPixel.h>

// LED strip settings
#define LED_PIN 19
#define NUM_LEDS 30   // change to your actual number

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

int darkThreshold = 30;

// ----------- PINS -----------
// Soil sensors
const int soilPins[4] = {1, 2, 3, 4};

// DHT11
#define DHTPIN 5
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Light sensor (analog)
const int lightPin = 6;

// Relay (pump)
const int relayPin = 10;

// ----------- CALIBRATION -----------
int dryValue = 3000;
int wetValue = 1200;

// ----------- FUNCTIONS -----------
int getMoisturePercent(int value) {
  int percent = map(value, dryValue, wetValue, 0, 100);
  if (percent > 100) percent = 100;
  if (percent < 0) percent = 0;
  return percent;
}

void setup() {
  Serial.begin(115200);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW);

  dht.begin();

  Serial.println("Smart Plant System Started 🌱");
}

void loop() {
  // ----------- SOIL -----------
  int soilRaw[4];
  int soilPercent[4];
  int sumPercent = 0;
  int dryCount = 0;

  for (int i = 0; i < 4; i++) {
    soilRaw[i] = analogRead(soilPins[i]);
    soilPercent[i] = getMoisturePercent(soilRaw[i]);

    sumPercent += soilPercent[i];

    if (soilPercent[i] < 30) {
      dryCount++;
    }
  }

  int avgSoil = sumPercent / 4;

  // ----------- DHT11 -----------
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // ----------- LIGHT -----------
  int lightRaw = analogRead(lightPin);
  int lightPercent = map(lightRaw, 0, 4095, 100, 0);

  // ----------- PRINT -----------
  Serial.println("------ DATA ------");

  for (int i = 0; i < 4; i++) {
    Serial.print("Soil ");
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.print(soilPercent[i]);
    Serial.println("%");
  }

  Serial.print("Avg Soil: ");
  Serial.print(avgSoil);
  Serial.println("%");

  Serial.print("Temp: ");
  Serial.print(temperature);
  Serial.println(" °C");

  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");

  Serial.print("Light: ");
  Serial.print(lightPercent);
  Serial.println("%");

  // ----------- GROW LIGHT (NEOPIXEL) -----------

int brightness = map(lightPercent, 0, 100, 255, 0);

// clamp
if (brightness < 0) brightness = 0;
if (brightness > 255) brightness = 255;

if (lightPercent < darkThreshold) {
  Serial.println("LED: GROW LIGHT ON 🌱");

  // reddish-white color
  uint8_t r = brightness;
  uint8_t g = brightness * 0.3;
  uint8_t b = brightness * 0.15;

  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(r, g, b));
  }
  strip.show();

} else {
  Serial.println("LED: OFF ☀️");

  strip.clear();
  strip.show();
}

  // ----------- WATER LOGIC -----------
  bool shouldWater = (dryCount >= 2);

  if (shouldWater) {
    Serial.println("STATUS: WATERING 💧");
    digitalWrite(relayPin, HIGH);
  } else {
    Serial.println("STATUS: OK 🌿");
    digitalWrite(relayPin, LOW);
  }

  Serial.println("------------------\n");

  delay(3000);
}