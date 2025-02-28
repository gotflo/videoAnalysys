//Colorrr

var canvas1 = document.getElementById('canvas');
var ctx1 = canvas1.getContext('2d');
var img1 = new Image();
img1.src = "../static/imgs/maillot.png"; // Remplacez par le chemin de votre image
img1.onload = function() {
    ctx1.drawImage(img1, 0, 0, canvas1.width, canvas1.height);
}

var canvas2 = document.getElementById('canvas1');
var ctx2 = canvas2.getContext('2d');
var img2 = new Image();
img2.src = "../static/imgs/maillot.png"; // Remplacez par le chemin de votre image
img2.onload = function() {
    ctx2.drawImage(img2, 0, 0, canvas2.width, canvas2.height);
}

var canvas3 = document.getElementById('canvas3');
var ctx3 = canvas3.getContext('2d');
var img3 = new Image();
img3.src = "../static/imgs/gant3.png"; // Remplacez par le chemin de votre image
img3.onload = function() {
    ctx3.drawImage(img3, 0, 0, canvas3.width, canvas3.height);
}
var canvas4 = document.getElementById('canvas4');
var ctx4 = canvas4.getContext('2d');
var img4 = new Image();
img4.src = "../static/imgs/gant3.png"; // Remplacez par le chemin de votre image
img4.onload = function() {
    ctx4.drawImage(img4, 0, 0, canvas4.width, canvas4.height);
}

canvas1.addEventListener('click', function() {
    document.getElementById('color-picker').click();
});

canvas2.addEventListener('click', function() {
    document.getElementById('color-picker1').click();
});

canvas3.addEventListener('click', function() {
    document.getElementById('color-picker3').click();
});

canvas4.addEventListener('click', function() {
    document.getElementById('color-picker4').click();
});


document.getElementById('color-picker').addEventListener('input', function() {
    applyColor(canvas1, ctx1, img1, this.value);
});

document.getElementById('color-picker1').addEventListener('input', function() {
    applyColor(canvas2, ctx2, img2, this.value);
});

document.getElementById('color-picker3').addEventListener('input', function() {
    applyColor(canvas3, ctx3, img3, this.value);
});

document.getElementById('color-picker4').addEventListener('input', function() {
    applyColor(canvas4, ctx4, img4, this.value);
});


function applyColor(canvas, ctx, img, color) {
    var [r, g, b] = hexToRgb(color);

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height); // Réinitialise l'image d'origine
    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var data = imageData.data;

    for (var i = 0; i < data.length; i += 4) {
        // Vérifie si le pixel est proche du blanc (maillot blanc)
        if (data[i] > 200 && data[i + 1] > 200 && data[i + 2] > 200) {
            data[i] = r;       // Rouge
            data[i + 1] = g;   // Vert
            data[i + 2] = b;   // Bleu
        }
    }

    ctx.putImageData(imageData, 0, 0);
}

function hexToRgb(hex) {
    var bigint = parseInt(hex.slice(1), 16);
    var r = (bigint >> 16) & 255;
    var g = (bigint >> 8) & 255;
    var b = bigint & 255;
    return [r, g, b];
}


//Decrement Increment


function previewVideo() {
            
    const videoFile = document.getElementById('input-file').files[0];
    const videoview = document.getElementById('video-view');
    const videoPreview = document.getElementById('video-preview');
    const button = document.getElementById('uploadLabel');

    const videoObjectURL = URL.createObjectURL(videoFile);
    videoPreview.src = videoObjectURL;
    videoview.style.display = 'none';
    videoPreview.style.display = 'block';
    button.style.display = 'block';
}



function decrementValue() {
    const input = document.getElementById('adjustable-input');
    input.value = Math.max(parseInt(input.value) - 1, parseInt(input.min));

  }

function incrementValue() {
    const input = document.getElementById('adjustable-input');
    input.value = Math.min(parseInt(input.value) + 1, parseInt(input.max));

}

function decrementValue1() {
   
    const input1 = document.getElementById('adjustable-input1');
    input1.value = Math.max(parseInt(input1.value) - 1, parseInt(input1.min));


}

function incrementValue1() {
    
    const input1 = document.getElementById('adjustable-input1');
    input1.value = Math.min(parseInt(input1.value) + 1, parseInt(input1.max));


}

function decrementValue2() {
   
    const input2 = document.getElementById('adjustable-input2');
    input2.value = Math.max(parseInt(input2.value) - 1, parseInt(input2.min));
}

function incrementValue2() {
  
    const input2 = document.getElementById('adjustable-input2');
    input2.value = Math.min(parseInt(input2.value) + 1, parseInt(input2.max));
}