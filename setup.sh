mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"julien.ract@sfr.fr\"\n\
" > ~/.streamlit/credentials.toml
#renaud.gantois@gmail.com
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
