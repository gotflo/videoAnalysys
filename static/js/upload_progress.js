/* Upload Files */


document.querySelector("html").classList.add("js");

var fileInput = document.querySelector(".uploadBtnP"),
  button = document.querySelector(".uploadLabelP"),
  the_return = document.querySelector(".file-return");

  var fileInput1 = document.querySelector(".uploadBtn1"),
  button1 = document.querySelector(".uploadLabel1"),
  the_return1 = document.querySelector(".file-return1");

  var fileInput2 = document.querySelector(".uploadBtn2"),
  button2 = document.querySelector(".uploadLabel2"),
  the_return2 = document.querySelector(".file-return2");


button.addEventListener("keydown", function (event) {
  if (event.keyCode == 13 || event.keyCode == 32) {
    fileInput.focus();
  }
});
button.addEventListener("click", function (event) {
  fileInput.focus();
  return false;
});
fileInput.addEventListener("change", function (event) {
  the_return.innerHTML = this.value;
});
//upload 1
button1.addEventListener("keydown", function (event) {
  if (event.keyCode == 13 || event.keyCode == 32) {
    fileInput1.focus();
  }
});
button1.addEventListener("click", function (event) {
  fileInput1.focus();
  return false;
});
fileInput1.addEventListener("change", function (event) {
  the_return1.innerHTML = this.value;
});
//upload 2
button2.addEventListener("keydown", function (event) {
  if (event.keyCode == 13 || event.keyCode == 32) {
    fileInput2.focus();
  }
});
button2.addEventListener("click", function (event) {
  fileInput2.focus();
  return false;
});
fileInput2.addEventListener("change", function (event) {
  the_return2.innerHTML = this.value;
});


/* Progress bar */

function loading() {
   
    // Affiche la barre de progression
    var progressBar = document.querySelector(".popup-progress");
    progressBar.style.display = "block";
  
    // Simulation de traitement de formulaire
    simulateFormProcessing();
  };
  
  function simulateFormProcessing() {
    var progressBar = document.querySelector(".bar");
    var progressCounter = document.querySelector(".counter");
    var width = 1;
    var interval = setInterval(function() {
      if (width >= 100) {
        clearInterval(interval);
        // Une fois le traitement terminé, masquez la barre de progression
        var progressBarContainer = document.querySelector(".popup-progress");
        progressBarContainer.style.display = "none";
      
      } else {
        
        width++;
        progressCounter.textContent= width + "%";
        progressBar.style.width = width + "%";
      }
    }, 1450); // Vitesse de progression de la barre en millisecondes
  }
  