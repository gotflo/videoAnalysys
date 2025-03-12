let headerListVisible = false; // Variable pour suivre l'état des styles de l'élément .header-list

const addStylesToHeaderList = () => {
  const headerList = document.querySelector('.header-list');
  headerList.style.display = 'flex';
  headerList.style.flexDirection = 'column';
  headerList.style.position = 'absolute';
  headerList.style.zIndex = '1000';
  headerList.style.left = '40%';
  headerList.style.backgroundColor = 'white';
  headerList.style.color='rgb(0, 247, 255)'
  headerListVisible = true; // Met à jour la variable de statut pour indiquer que les styles sont actuellement appliqués
}

document.addEventListener("DOMContentLoaded", function () {
    let images = [
        "../static/imgs/peoplesoccer.jpg",
        "../static/imgs/trainer1.jpg",
        "../static/imgs/trainer2.jpg"
    ];

    let banner = document.querySelector(".banner");
    let index = 0;

    function changeImage() {
        banner.style.backgroundImage = `url('${images[index]}')`;
        index = (index + 1) % images.length;
    }

    // Définir l'image de départ
    banner.style.backgroundImage = `url('${images[0]}')`;

    // Changer l'image toutes les 3 secondes
    setInterval(changeImage, 3000);
});

document.addEventListener("DOMContentLoaded", function () {
    let cards = document.querySelectorAll(".pricing-card");

    cards.forEach(card => {
        card.addEventListener("click", function () {
            // Supprime la classe "selected" de toutes les cartes
            cards.forEach(c => c.classList.remove("selected"));

            // Ajoute la classe "selected" à la carte cliquée
            this.classList.add("selected");
        });
    });
});



const removeStylesFromHeaderList = () => {
  const headerList = document.querySelector('.header-list');
  headerList.style.display = ''; // Remise à la valeur par défaut
  headerList.style.flexDirection = ''; // Remise à la valeur par défaut
  headerList.style.position = ''; // Remise à la valeur par défaut
  headerList.style.zIndex = ''; // Remise à la valeur par défaut
  headerList.style.left = ''; // Remise à la valeur par défaut
  headerList.style.backgroundColor = ''; // Remise à la valeur par défaut
  headerListVisible = false; // Met à jour la variable de statut pour indiquer que les styles ne sont pas actuellement appliqués
}

const toggleStylesForHeaderList = () => {
  if (headerListVisible) {
    removeStylesFromHeaderList(); // Si les styles sont actuellement appliqués, les retire
  } else {
    addStylesToHeaderList(); // Sinon, les ajoute
  }
}

const svg = document.getElementById('svg');
svg.addEventListener("click", toggleStylesForHeaderList); // Utilise la fonction de bascule lors du clic sur l'élément SVG

const close = document.getElementById("close");
close.addEventListener("click", removeStylesFromHeaderList); // Retire les styles lorsque l'élément avec l'ID "close" est cliqué

// let filename = document.getElementsByName('filename')[0]; // Obtenez le premier élément avec le nom 'filename'
// if (filename.value !== '') { // Vérifiez la valeur de l'élément filename
//   let header = document.querySelector('header'); // Sélectionnez l'élément header
//   header.style.marginBottom = '15px'; // Modifiez la marge inférieure de l'élément header
// }


