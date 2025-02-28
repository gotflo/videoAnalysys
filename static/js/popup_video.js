
function agrandirImgHome() {
    document.getElementById('popup-imgHome').style.display = 'block';
   
}

function fermerImg() {
    document.getElementById('popup-imgHome').style.display = 'none';
    document.getElementById('popup-imgAway').style.display = 'none';
    
}

function agrandirImgAway() {
    document.getElementById('popup-imgAway').style.display = 'block';
   
}


function afficherSimulation(){
    document.getElementById('match_simulation').style.display = 'block';
    document.getElementById('epv').style.display = 'none';
    document.getElementById('player_tracking').style.display = 'none';
    document.getElementById('pass_analysis').style.display = 'none';
    document.getElementById('sprint_visualisation').style.display = 'none';
    document.getElementById('xg').style.display = 'none';
    document.getElementById('peatch_control').style.display = 'none';
}
function afficherTracking(){
    document.getElementById('player_tracking').style.display = 'block';
    document.getElementById('epv').style.display = 'none';
    document.getElementById('match_simulation').style.display = 'none';
    document.getElementById('pass_analysis').style.display = 'none';
    document.getElementById('sprint_visualisation').style.display = 'none';
    document.getElementById('xg').style.display = 'none';
    document.getElementById('peatch_control').style.display = 'none';

}

function afficherPass(){
    document.getElementById('pass_analysis').style.display = 'block';
    document.getElementById('epv').style.display = 'none';
    document.getElementById('player_tracking').style.display = 'none';
    document.getElementById('match_simulation').style.display = 'none';
    document.getElementById('sprint_visualisation').style.display = 'none';
    document.getElementById('xg').style.display = 'none';
    document.getElementById('peatch_control').style.display = 'none';

}

function afficherSprint(){
    document.getElementById('sprint_visualisation').style.display = 'block';
    document.getElementById('epv').style.display = 'none';
    document.getElementById('pass_analysis').style.display = 'none';
    document.getElementById('player_tracking').style.display = 'none';
    document.getElementById('match_simulation').style.display = 'none';
    document.getElementById('xg').style.display = 'none';
    document.getElementById('peatch_control').style.display = 'none';

}

function afficherEpv(){
    document.getElementById('epv').style.display = 'block';
    document.getElementById('sprint_visualisation').style.display = 'none';
    document.getElementById('pass_analysis').style.display = 'none';
    document.getElementById('player_tracking').style.display = 'none';
    document.getElementById('match_simulation').style.display = 'none';
    document.getElementById('xg').style.display = 'none';
    document.getElementById('peatch_control').style.display = 'none';

}



function afficherXG(){
    document.getElementById('xg').style.display = 'block';
    document.getElementById('epv').style.display = 'none';
    document.getElementById('sprint_visualisation').style.display = 'none';
    document.getElementById('pass_analysis').style.display = 'none';
    document.getElementById('player_tracking').style.display = 'none';
    document.getElementById('match_simulation').style.display = 'none';
    document.getElementById('peatch_control').style.display = 'none';

}

function afficherPeatchControl(){
    document.getElementById('peatch_control').style.display = 'block';
    document.getElementById('epv').style.display = 'none';
    document.getElementById('sprint_visualisation').style.display = 'none';
    document.getElementById('pass_analysis').style.display = 'none';
    document.getElementById('player_tracking').style.display = 'none';
    document.getElementById('match_simulation').style.display = 'none';
    document.getElementById('xg').style.display = 'none';

}


// Agrandir l'image des epv,player_tracking,pass_analysis,sprint_visualisation


function agrandirImgplayerTracking() {
    agrandirImage('tracking_img', 'Tracking');
}

function agrandirImgPass() {
    agrandirImage('pass_analysis_img', 'Pass Analysis');
}

function agrandirImgSprint() {
    agrandirImage('img_sprint', 'Sprint');
}

function agrandirImgEPV() {
    agrandirImage('epv_img', 'EPV');
}

function agrandirImgEXG() {
    agrandirImage('xg_img', 'xG');
}

function agrandirImage(imgId, altText) {
    var modal = document.getElementById("myModal");
    var img = document.getElementById(imgId);
    var modalImg = document.getElementById("img01");
    var captionText = document.getElementById("caption");

    modal.style.display = "block";
    modalImg.src = img.src;
    captionText.innerHTML = altText;
}

function closeModal() {
    var modal = document.getElementById("myModal");
    modal.style.display = "none";
}