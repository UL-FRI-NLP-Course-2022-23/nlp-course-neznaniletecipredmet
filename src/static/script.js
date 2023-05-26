document.getElementById("myForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent form submission

    // Get the text input value
    var textInput = document.getElementById("textInput").value;

    // Send an AJAX request to the server
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/process", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            console.log("Response received: " + xhr.responseText);
        }
        location.reload ? location.reload() : location = location;
    };
    xhr.send(JSON.stringify({ text: textInput }));
});

window.onload = function() {
  var sketchSelector = document.getElementById("sketchSelector");
  var fileNames = ['Babica_in_njeni_štirje_vnučki', 'Babica_in_trije_vnučki', 'Bela_kača_s_kronico', 'Celovški_zmaj', 'Deklica_in_pasjeglavci', 'Dolgouhec_in_medved', 'Dva_brata', 'Dva_lenuha', 'Ježek',
    'Jožkove_sanje', 'Kaj_delajo_muce,_ko_spijo', 'Kako_je_Barica_prevarala_čarovnico', 'Kako_so_nastali_krofi', 'Kako_so_pulili_repo', 'Kako_so_spominčice_dobile_ime',
    'Kam_gre_staro_leto_in_od_kod_pride_novo', 'Kdo_bo_popravil_Luno_', 'Ko_bi_bilo_vedno_tako', 'Kraljevič_in_Lepa_Vida', 'Kurent', 'Lisica,_volk_in_medved', 'Lisica_in_jež', 'Lisica_in_njene_mlade',
    'Lisica_in_vogelk', 'Lucifer_se_ženi', 'Mlinar,_njegov_kuhar_in_kralj', 'Mož_in_medved', 'Nace', 'Najdražji_zaklad', 'Nehvaležna_brata', 'Nevoščljivi_zdravnik', 'Nožek',
    'Od_kdaj_ima_zajček_kratek_rep', 'O_dvanajstih_bratih_in_sestrah', 'O_fantu_z_mačko', 'O_petelinu_in_zmaju', 'O_zlatih_jabolkih', 'Oče_je_upil_zvonec,_mati_pa_lonec', 'Palček',
    'Pastir_s_čudežnim_kravjim_rogom', 'Pastirček', 'Pastirček_in_čarovnikova_hči', 'Petelin_in_zmaj', 'Pogašeni_zmaj', 'Pravljica_o_žabi', 'Sedem_let_pri_beli_kači', 'Sin_jež', 'Sirotica', 'Smrt_in_čevljar',
    'Speči_zajec', 'Tea_in_čarobna_travica', 'Topli_potok', 'Trije_velikani', 'Tri_botre_lisičice', 'Tri_revne_deklice', 'Užaljeni_velikan', 'Vikin_čarobni_kaktus', 'Zakaj_imajo_krave_roge',
    'Zakaj_sta_Luna_in_Sonce_na_nebu_posebej', 'Zakleti_mlin', 'Začarani_grad_in_medved', 'Začaran_grad_in_medved', 'Zdravilno_jabolko', 'Zlata_gora', 'Zlata_ptica', 'Zlata_ribica',
    'Zlatolaska', 'Zlatorog', 'Zmaj_v_Peci', 'Zmaj_v_Postojnski_jami', 'Zviti_Martin', 'Času_primerna_pravljica', 'Čisto_prava_mamica', 'Čudežni_studenec']
  fileNames.forEach(function(fileName) {
    var optionElement = document.createElement("option");
    optionElement.value = fileName;
    optionElement.textContent = fileName;
    sketchSelector.appendChild(optionElement);
  });
};

document.getElementById("example").addEventListener("click", function() {

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/example", true);
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            console.log("Response received: " + xhr.responseText);
        }
        location.reload();
        console.log("reloaded");
    };
    xhr.send();
});
