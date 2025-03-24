#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdbool.h>
#include <stdint.h>

// --- Pin Definitions ---
// BCD outputs for the 7447 decoder (Arduino D2-D5).
#define BCD_A    PD2
#define BCD_B    PD3
#define BCD_C    PD4
#define BCD_D    PD5

// Digit enable pins for multiplexing the 7-seg displays (Arduino D6-D11).
// Mapping: D6 -> DIGIT_S1, D7 -> DIGIT_S2, D8 -> DIGIT_S3, D9 -> DIGIT_S4, D10 -> DIGIT_S5, D11 -> DIGIT_S6.
#define DIGIT_S1 PD6
#define DIGIT_S2 PD7
#define DIGIT_S3 PB0
#define DIGIT_S4 PB1
#define DIGIT_S5 PB2
#define DIGIT_S6 PB3

// Button pins (using internal pull-ups).
// On the ATmega328, A0-A3 map to PC0-PC3 and A5 maps to PC5.
#define BTN_HH   PC0   // HH increment button
#define BTN_MM   PC1   // MM increment button
#define BTN_SS   PC2   // SS increment button
#define BTN_PAUSE PC3  // Pause/Play toggle button (short press toggles pause; long press resets active mode)
#define BTN_TIMER PC5  // Timer mode toggle button

// --- Configuration Constants ---
const uint32_t longPressThreshold = 2000; // 2 seconds for continuous increment
const uint32_t repeatDelay        = 200;    // 200 ms delay for auto-repeat
const uint32_t pauseResetHoldTime = 5000;   // 5 seconds for reset when held

// --- Global Timekeeping Variables ---
// Two independent time states: clock mode (count up) and timer mode (count down).
volatile uint32_t timer_millis = 0;  // Updated by Timer0 ISR

// Clock mode time (count up).
uint8_t clockHours = 12, clockMinutes = 20, clockSeconds = 32;
// Timer mode time (count down).
uint8_t timerHours = 0, timerMinutes = 0, timerSeconds = 0;
uint32_t lastMillis = 0;

// --- Global Flags ---
volatile bool timerMode = false;   // false = clock mode; true = timer mode
volatile bool clockPaused = false; // Pausing flag for clock mode
volatile bool timerPaused = false; // Pausing flag for timer mode

// --- Timer0 Compare Match Interrupt Service Routine ---
// Triggered every 1 ms.
ISR(TIMER0_COMPA_vect) {
    timer_millis++;
}

// Returns the number of milliseconds since the timer started.
uint32_t millis(void) {
    uint32_t ms;
    cli();
    ms = timer_millis;
    sei();
    return ms;
}

void setup(void) {
    // Configure BCD output pins (D2-D5) as outputs.
    DDRD |= (1 << BCD_A) | (1 << BCD_B) | (1 << BCD_C) | (1 << BCD_D);
    
    // Configure digit enable pins.
    DDRD |= (1 << DIGIT_S1) | (1 << DIGIT_S2);
    DDRB |= (1 << DIGIT_S3) | (1 << DIGIT_S4) | (1 << DIGIT_S5) | (1 << DIGIT_S6);
    
    // Configure buttons (A0-A3, A5) as inputs with internal pull-ups.
    DDRC &= ~((1 << BTN_HH) | (1 << BTN_MM) | (1 << BTN_SS) | (1 << BTN_PAUSE) | (1 << BTN_TIMER));
    PORTC |= ((1 << BTN_HH) | (1 << BTN_MM) | (1 << BTN_SS) | (1 << BTN_PAUSE) | (1 << BTN_TIMER));
    
    // Initialize output pins to low.
    PORTD &= ~((1 << BCD_A) | (1 << BCD_B) | (1 << BCD_C) | (1 << BCD_D));
    PORTD &= ~((1 << DIGIT_S1) | (1 << DIGIT_S2));
    PORTB &= ~((1 << DIGIT_S3) | (1 << DIGIT_S4) | (1 << DIGIT_S5) | (1 << DIGIT_S6));
    
    // Set up Timer0 for a 1ms tick.
    TCCR0A = (1 << WGM01);              // CTC mode.
    TCCR0B = (1 << CS01) | (1 << CS00);   // Prescaler 64.
    OCR0A = 249;                        // 1ms period.
    TIMSK0 |= (1 << OCIE0A);            // Enable Timer0 Compare Match A interrupt.
}

// --- Display Functions ---
// Displays one digit on the 7447 by setting BCD outputs and enabling the correct digit.
void displayDigit(uint8_t digit, uint8_t position) {
    // Set BCD outputs.
    if (digit & 0x01)
        PORTD |= (1 << BCD_A);
    else
        PORTD &= ~(1 << BCD_A);
    
    if (digit & 0x02)
        PORTD |= (1 << BCD_B);
    else
        PORTD &= ~(1 << BCD_B);
    
    if (digit & 0x04)
        PORTD |= (1 << BCD_C);
    else
        PORTD &= ~(1 << BCD_C);
    
    if (digit & 0x08)
        PORTD |= (1 << BCD_D);
    else
        PORTD &= ~(1 << BCD_D);
    
    // Turn off all digit enable pins.
    PORTD &= ~((1 << DIGIT_S1) | (1 << DIGIT_S2));
    PORTB &= ~((1 << DIGIT_S3) | (1 << DIGIT_S4) | (1 << DIGIT_S5) | (1 << DIGIT_S6));
    
    // Enable the specific digit based on its position.
    switch(position) {
        case 0: PORTB |= (1 << DIGIT_S6); break; // Hours tens (Arduino D11)
        case 1: PORTB |= (1 << DIGIT_S5); break; // Hours units (Arduino D10)
        case 2: PORTB |= (1 << DIGIT_S4); break; // Minutes tens (Arduino D9)
        case 3: PORTB |= (1 << DIGIT_S3); break; // Minutes units (Arduino D8)
        case 4: PORTD |= (1 << DIGIT_S2); break; // Seconds tens (Arduino D7)
        case 5: PORTD |= (1 << DIGIT_S1); break; // Seconds units (Arduino D6)
        default: break;
    }
    
    _delay_ms(2);  // Small delay for digit visibility.
    
    // Disable the digit after the delay.
    switch(position) {
        case 0: PORTB &= ~(1 << DIGIT_S6); break;
        case 1: PORTB &= ~(1 << DIGIT_S5); break;
        case 2: PORTB &= ~(1 << DIGIT_S4); break;
        case 3: PORTB &= ~(1 << DIGIT_S3); break;
        case 4: PORTD &= ~(1 << DIGIT_S2); break;
        case 5: PORTD &= ~(1 << DIGIT_S1); break;
        default: break;
    }
}

// Updates the display with the time from the active mode.
void updateDisplay(void) {
    if (timerMode) {
        displayDigit(timerHours / 10, 0);
        displayDigit(timerHours % 10, 1);
        displayDigit(timerMinutes / 10, 2);
        displayDigit(timerMinutes % 10, 3);
        displayDigit(timerSeconds / 10, 4);
        displayDigit(timerSeconds % 10, 5);
    } else {
        displayDigit(clockHours / 10, 0);
        displayDigit(clockHours % 10, 1);
        displayDigit(clockMinutes / 10, 2);
        displayDigit(clockMinutes % 10, 3);
        displayDigit(clockSeconds / 10, 4);
        displayDigit(clockSeconds % 10, 5);
    }
}

// Checks buttons and updates time values for the active mode.
// Increment buttons update HH, MM, SS. The pause button toggles pause (or resets time if held).
// The timer mode button toggles between timer and clock modes.
void checkButtons(void) {
    uint32_t currentMillis = millis();
    
    // --- Hour Button (BTN_HH, PC0) ---
    static uint32_t hhHoldStart = 0, hhLastRepeat = 0;
    bool buttonHourState = (PINC & (1 << BTN_HH)) != 0;
    if (!buttonHourState) { // Button pressed (active low)
        if (hhHoldStart == 0) {
            hhHoldStart = currentMillis;
            hhLastRepeat = currentMillis;
        } else if (currentMillis - hhHoldStart >= longPressThreshold &&
                   currentMillis - hhLastRepeat >= repeatDelay) {
            if (timerMode)
                timerHours = (timerHours + 1) % 24;
            else
                clockHours = (clockHours + 1) % 24;
            hhLastRepeat = currentMillis;
        }
    } else {
        if (hhHoldStart != 0 && currentMillis - hhHoldStart < longPressThreshold) {
            if (timerMode)
                timerHours = (timerHours + 1) % 24;
            else
                clockHours = (clockHours + 1) % 24;
        }
        hhHoldStart = 0;
        hhLastRepeat = 0;
    }
    
    // --- Minute Button (BTN_MM, PC1) ---
    static uint32_t mmHoldStart = 0, mmLastRepeat = 0;
    bool buttonMinuteState = (PINC & (1 << BTN_MM)) != 0;
    if (!buttonMinuteState) {
        if (mmHoldStart == 0) {
            mmHoldStart = currentMillis;
            mmLastRepeat = currentMillis;
        } else if (currentMillis - mmHoldStart >= longPressThreshold &&
                   currentMillis - mmLastRepeat >= repeatDelay) {
            if (timerMode)
                timerMinutes = (timerMinutes + 1) % 60;
            else
                clockMinutes = (clockMinutes + 1) % 60;
            mmLastRepeat = currentMillis;
        }
    } else {
        if (mmHoldStart != 0 && currentMillis - mmHoldStart < longPressThreshold) {
            if (timerMode)
                timerMinutes = (timerMinutes + 1) % 60;
            else
                clockMinutes = (clockMinutes + 1) % 60;
        }
        mmHoldStart = 0;
        mmLastRepeat = 0;
    }
    
    // --- Second Button (BTN_SS, PC2) ---
    static uint32_t ssHoldStart = 0, ssLastRepeat = 0;
    bool buttonSecondState = (PINC & (1 << BTN_SS)) != 0;
    if (!buttonSecondState) {
        if (ssHoldStart == 0) {
            ssHoldStart = currentMillis;
            ssLastRepeat = currentMillis;
        } else if (currentMillis - ssHoldStart >= longPressThreshold &&
                   currentMillis - ssLastRepeat >= repeatDelay) {
            if (timerMode)
                timerSeconds = (timerSeconds + 1) % 60;
            else
                clockSeconds = (clockSeconds + 1) % 60;
            ssLastRepeat = currentMillis;
        }
    } else {
        if (ssHoldStart != 0 && currentMillis - ssHoldStart < longPressThreshold) {
            if (timerMode)
                timerSeconds = (timerSeconds + 1) % 60;
            else
                clockSeconds = (clockSeconds + 1) % 60;
        }
        ssHoldStart = 0;
        ssLastRepeat = 0;
    }
    
    // --- Pause/Play Button (BTN_PAUSE, PC3) ---
    static uint32_t pauseHoldStart = 0;
    bool buttonPauseState = (PINC & (1 << BTN_PAUSE)) != 0;
    if (!buttonPauseState) {
        if (pauseHoldStart == 0)
            pauseHoldStart = currentMillis;
        else if (currentMillis - pauseHoldStart >= pauseResetHoldTime) {
            if (timerMode) {
                timerHours = timerMinutes = timerSeconds = 0;
                timerPaused = true;
            } else {
                clockHours = clockMinutes = clockSeconds = 0;
                clockPaused = true;
            }
            pauseHoldStart = currentMillis;
        }
    } else {
        if (pauseHoldStart != 0 && currentMillis - pauseHoldStart < pauseResetHoldTime) {
            if (timerMode)
                timerPaused = !timerPaused;
            else
                clockPaused = !clockPaused;
        }
        pauseHoldStart = 0;
    }
    
    // --- Timer Mode Button (BTN_TIMER, PC5) ---
    static bool lastTimerState = true;
    bool buttonTimerState = (PINC & (1 << BTN_TIMER)) != 0;
    if (lastTimerState == false && buttonTimerState == true) {
        timerMode = !timerMode;
    }
    lastTimerState = buttonTimerState;
}

int main(void) {
    setup();
    sei();  // Enable global interrupts.
    
    while (1) {
        uint32_t currentMillis = millis();
        if ((currentMillis - lastMillis) >= 1000) {
            lastMillis = currentMillis;
            
            // Update clock mode time (count up) regardless of active mode.
            if (!clockPaused) {
                clockSeconds++;
                if (clockSeconds >= 60) {
                    clockSeconds = 0;
                    clockMinutes++;
                }
                if (clockMinutes >= 60) {
                    clockMinutes = 0;
                    clockHours++;
                }
                if (clockHours >= 24)
                    clockHours = 0;
            }
            
            // Update timer mode time (count down) regardless of active mode.
            if (!timerPaused) {
                if (timerHours == 0 && timerMinutes == 0 && timerSeconds == 0) {
                    // Timer finished; hold at 0.
                } else {
                    if (timerSeconds > 0)
                        timerSeconds--;
                    else {
                        timerSeconds = 59;
                        if (timerMinutes > 0)
                            timerMinutes--;
                        else {
                            timerMinutes = 59;
                            if (timerHours > 0)
                                timerHours--;
                        }
                    }
                }
            }
            
            // --- Begin Boolean Logic Example ---
            uint8_t activeSeconds = timerMode ? timerSeconds : clockSeconds;
            bool W = (activeSeconds & 0x08) != 0;
            bool X = (activeSeconds & 0x04) != 0;
            bool Y = (activeSeconds & 0x02) != 0;
            bool Z = (activeSeconds & 0x01) != 0;
            
            bool A_logic = false;
            bool B_logic = (W && !X && !Y && !Z) ||
                           (!W && X && !Y && !Z) ||
                           (W && !X && Y && !Z) ||
                           (!W && X && Y && !Z);
            bool C_logic = (W && X && !Y && !Z) ||
                           (!W && !X && Y && !Z) ||
                           (W && !X && Y && !Z) ||
                           (!W && X && Y && !Z);
            bool D_logic = (W && X && Y && !Z) ||
                           (!W && !X && !Y && Z);
            A_logic = B_logic;
            B_logic = C_logic;
            C_logic = D_logic;
            D_logic = A_logic;
            // --- End Boolean Logic Example ---
        }
        
        checkButtons();
        updateDisplay();
    }
    
    return 0;
}
