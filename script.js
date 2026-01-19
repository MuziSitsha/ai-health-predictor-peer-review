// Ìºü Initialize EmailJS with your PUBLIC KEY
// ‚≠êÔ∏è REPLACE "YOUR_PUBLIC_KEY_HERE" with your actual key from EmailJS dashboard
(function() {
    emailjs.init("YOUR_PUBLIC_KEY_HERE"); // ‚ö†Ô∏è REPLACE THIS WITH YOUR ACTUAL KEY!
})();

// ÌæØ DOM Elements
const welcomeScreen = document.getElementById('welcomeScreen');
const cardContainer = document.getElementById('cardContainer');
const openBtn = document.getElementById('openBtn');
const card = document.querySelector('.card');
const balloonsContainer = document.getElementById('balloonsContainer');
const confettiContainer = document.getElementById('confettiContainer');
const yesBtn = document.getElementById('yesBtn');
const noBtn = document.getElementById('noBtn');
const responseForm = document.getElementById('responseForm');
const userNameInput = document.getElementById('userName');
const submitResponse = document.getElementById('submitResponse');
const successMessage = document.getElementById('successMessage');
const options = document.getElementById('options');
const footer = document.getElementById('footer');

// Ìæà Balloon emojis and colors
const balloonEmojis = ['Ìæà', '‚ù§Ô∏è', 'Ì≤ñ', 'Ì≤ï', 'Ì≤ó', 'Ì≤ì', 'Ì≤û', 'Ì≤ù', 'ÌæÄ', 'ÌæÅ'];
const balloonColors = [
    '#ff4d6d', '#ff8fab', '#ff758c', '#ffafcc', 
    '#c9184a', '#ffccd5', '#ff477e', '#ff5c8a',
    '#ff6b9d', '#ff85a1'
];

// Ìæâ Confetti emojis
const confettiEmojis = ['‚ú®', 'Ìºü', 'Ì≤´', 'Ìæä', 'Ìæâ', 'Ì≤ñ', 'Ì≤ï', 'Ì≤ó', 'Ì≤ì', 'ÌæÄ'];

// Ìºü Open card button
openBtn.addEventListener('click', function() {
    welcomeScreen.style.opacity = '0';
    setTimeout(() => {
        welcomeScreen.style.display = 'none';
        cardContainer.style.display = 'block';
        setTimeout(() => {
            document.querySelector('.container').classList.add('card-opened');
            createBalloons(20);
            createFloatingHearts(15);
        }, 300);
    }, 500);
});

// Ì≤ñ Card click to open
card.addEventListener('click', function() {
    if (!document.querySelector('.container').classList.contains('card-opened')) {
        document.querySelector('.container').classList.add('card-opened');
        createBalloons(15);
        createFloatingHearts(10);
    }
});

// ‚úÖ Yes button click
yesBtn.addEventListener('click', function() {
    options.style.opacity = '0';
    setTimeout(() => {
        options.style.display = 'none';
        responseForm.style.display = 'block';
        setTimeout(() => {
            responseForm.style.opacity = '1';
        }, 50);
    }, 500);
    
    createConfetti(150);
    createBalloons(25);
    createFloatingHearts(20);
    footer.style.opacity = '0';
});

// ‚ùå No button click
let noClickCount = 0;
const noButtonTexts = [
    'Ì¥î Hmm... let me think',
    'Ì∏è Are you sure?',
    'Ì∏¢ Breaking my heart...',
    'Ìµ∫ Pretty please?',
    'Ì∏≠ Don\'t make me sad!',
    'Ì∏ò Just kidding! YES! Ì≤ñ'
];

noBtn.addEventListener('click', function() {
    const maxX = window.innerWidth - noBtn.offsetWidth;
    const maxY = window.innerHeight - noBtn.offsetHeight;
    const x = Math.random() * maxX;
    const y = Math.random() * maxY;
    
    noBtn.style.transition = 'all 0.5s';
    noBtn.style.position = 'fixed';
    noBtn.style.left = `${x}px`;
    noBtn.style.top = `${y}px`;
    noBtn.style.transform = `rotate(${Math.random() * 20 - 10}deg)`;
    
    if (noClickCount < noButtonTexts.length) {
        noBtn.textContent = noButtonTexts[noClickCount];
        noClickCount++;
        createSmallHearts(5);
    } else {
        noBtn.style.display = 'none';
        options.innerHTML = `
            <div style="animation: fadeIn 1s ease;">
                <h3 style="color: #ff4d6d; font-size: 2rem;">Ìæ≠ You win!</h3>
                <button class="btn yes-btn" onclick="location.reload()">
                    Ì≤ñ YES, I\'LL BE YOUR VALENTINE! Ì≤ñ
                </button>
            </div>
        `;
        createConfetti(100);
    }
});

// ‚úâÔ∏è Submit response
submitResponse.addEventListener('click', function() {
    const userName = userNameInput.value.trim();
    
    if (!userName) {
        userNameInput.style.animation = 'shake 0.5s';
        userNameInput.style.borderColor = '#ff4d6d';
        userNameInput.placeholder = 'Ì≤ñ Please enter your name! Ì≤ñ';
        setTimeout(() => {
            userNameInput.style.animation = '';
        }, 500);
        return;
    }
    
    // Send email using EmailJS
    sendEmailResponse(userName, 'YES Ì≤ñ');
    
    responseForm.style.opacity = '0';
    setTimeout(() => {
        responseForm.style.display = 'none';
        successMessage.style.display = 'block';
        setTimeout(() => {
            successMessage.style.opacity = '1';
        }, 50);
    }, 500);
    
    createConfetti(300);
    createBalloons(50);
    createFloatingHearts(30);
    
    footer.innerHTML = `<p>Ì≤ñ ${userName} said YES! Ì≤ñ</p>`;
    footer.style.opacity = '1';
});

// Ìæà Function to create balloons
function createBalloons(count) {
    for (let i = 0; i < count; i++) {
        const balloon = document.createElement('div');
        balloon.className = 'balloon';
        const size = Math.random() * 40 + 50;
        const color = balloonColors[Math.floor(Math.random() * balloonColors.length)];
        const emoji = balloonEmojis[Math.floor(Math.random() * balloonEmojis.length)];
        const left = Math.random() * 100;
        const duration = Math.random() * 15 + 15;
        const delay = Math.random() * 3;
        
        balloon.style.width = `${size}px`;
        balloon.style.height = `${size * 1.1}px`;
        balloon.style.background = color;
        balloon.style.left = `${left}%`;
        balloon.style.animation = `balloonRise ${duration}s linear ${delay}s forwards`;
        balloon.textContent = emoji;
        balloon.style.fontSize = `${size * 0.4}px`;
        
        balloonsContainer.appendChild(balloon);
        
        setTimeout(() => {
            balloon.style.opacity = '0';
            setTimeout(() => balloon.remove(), 1000);
        }, (duration + delay) * 1000);
    }
}

// ‚ú® Function to create confetti
function createConfetti(count) {
    for (let i = 0; i < count; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        const emoji = confettiEmojis[Math.floor(Math.random() * confettiEmojis.length)];
        const size = Math.random() * 20 + 15;
        const left = Math.random() * 100;
        const duration = Math.random() * 3 + 2;
        
        confetti.textContent = emoji;
        confetti.style.left = `${left}%`;
        confetti.style.fontSize = `${size}px`;
        confettiContainer.appendChild(confetti);
        
        const animation = confetti.animate([
            { transform: 'translateY(0) rotate(0deg)', opacity: 1 },
            { transform: `translateY(${window.innerHeight}px) rotate(720deg)`, opacity: 0 }
        ], {
            duration: duration * 1000,
            easing: 'cubic-bezier(0.215, 0.610, 0.355, 1)'
        });
        
        animation.onfinish = () => confetti.remove();
    }
}

// Ì≤ñ Function to create floating hearts
function createFloatingHearts(count) {
    const heartEmojis = ['‚ù§Ô∏è', 'Ì≤ñ', 'Ì≤ï', 'Ì≤ó', 'Ì≤ì', 'Ì≤û', 'Ì≤ù'];
    for (let i = 0; i < count; i++) {
        const heart = document.createElement('div');
        heart.style.position = 'fixed';
        heart.style.fontSize = `${Math.random() * 30 + 20}px`;
        heart.style.left = `${Math.random() * 100}%`;
        heart.style.top = `${Math.random() * 100}%`;
        heart.style.opacity = '0';
        heart.style.zIndex = '6';
        heart.textContent = heartEmojis[Math.floor(Math.random() * heartEmojis.length)];
        document.body.appendChild(heart);
        
        heart.animate([
            { transform: 'translateY(0) scale(0)', opacity: 0 },
            { transform: 'translateY(-20px) scale(1)', opacity: 1 },
            { transform: 'translateY(-100px) scale(0.5)', opacity: 0 }
        ], {
            duration: 2000,
            easing: 'cubic-bezier(0.215, 0.610, 0.355, 1)'
        }).onfinish = () => heart.remove();
    }
}

// ÌæØ Helper functions
function createSmallHearts(count) {
    for (let i = 0; i < count; i++) {
        setTimeout(() => {
            const heart = document.createElement('div');
            heart.textContent = 'Ì≤ñ';
            heart.style.position = 'fixed';
            heart.style.fontSize = '20px';
            heart.style.left = `${Math.random() * 100}%`;
            heart.style.top = `${Math.random() * 100}%`;
            heart.style.zIndex = '9';
            document.body.appendChild(heart);
            
            heart.animate([
                { transform: 'translateY(0) scale(1)', opacity: 1 },
                { transform: 'translateY(-50px) scale(0.5)', opacity: 0 }
            ], {
                duration: 1000,
                easing: 'ease-out'
            }).onfinish = () => heart.remove();
        }, i * 100);
    }
}

function createFloatingEmoji(emoji, x, y) {
    const floatEmoji = document.createElement('div');
    floatEmoji.textContent = emoji;
    floatEmoji.style.position = 'fixed';
    floatEmoji.style.left = `${x}px`;
    floatEmoji.style.top = `${y}px`;
    floatEmoji.style.fontSize = '30px';
    floatEmoji.style.zIndex = '9';
    document.body.appendChild(floatEmoji);
    
    floatEmoji.animate([
        { transform: 'translateY(0) scale(1)', opacity: 1 },
        { transform: 'translateY(-100px) scale(0.5)', opacity: 0 }
    ], {
        duration: 1500,
        easing: 'ease-out'
    }).onfinish = () => floatEmoji.remove();
}

// Ì≤å Function to send email response
function sendEmailResponse(name, response) {
    // ‚ö†Ô∏è REPLACE THESE WITH YOUR EMAILJS DETAILS:
    const templateParams = {
        to_name: 'Muzi',
        from_name: name || 'Rikitai',
        response: response,
        date: new Date().toLocaleDateString('en-US', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        }),
        time: new Date().toLocaleTimeString(),
        special_message: 'RIKITAI SAID YES TO BEING YOUR VALENTINE!',
        celebration: 'Time to celebrate!'
    };
    
    // ‚úÖ Send email using EmailJS
    // ‚ö†Ô∏è REPLACE 'YOUR_TEMPLATE_ID' with your actual template ID
    // ‚ö†Ô∏è Service ID should be: service_soft9xw (from your screenshot)
    emailjs.send('service_soft9xw', 'YOUR_TEMPLATE_ID', templateParams)
        .then(function(response) {
            console.log('Ì≤å Email sent successfully!', response.status, response.text);
        })
        .catch(function(error) {
            console.error('‚ùå Failed to send email:', error);
        });
}

// Ìºü Add CSS animation for shake
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
`;
document.head.appendChild(style);

// ‚ú® Add initial animation on load
window.addEventListener('load', function() {
    createBalloons(8);
    createFloatingHearts(5);
});

// Ì≤´ Make the page extra special
document.body.addEventListener('click', function(e) {
    if (Math.random() > 0.7) {
        const heart = document.createElement('div');
        heart.textContent = 'Ì≤ñ';
        heart.style.position = 'fixed';
        heart.style.left = `${e.clientX}px`;
        heart.style.top = `${e.clientY}px`;
        heart.style.fontSize = '20px';
        heart.style.zIndex = '999';
        document.body.appendChild(heart);
        
        heart.animate([
            { transform: 'translateY(0) scale(1)', opacity: 1 },
            { transform: 'translateY(-30px) scale(0.5)', opacity: 0 }
        ], {
            duration: 1000,
            easing: 'ease-out'
        }).onfinish = () => heart.remove();
    }
});
