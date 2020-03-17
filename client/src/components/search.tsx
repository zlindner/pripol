import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import Analyze from '../assets/analyze.svg';

const Container = styled.div`
    width: 800px;
    display: flex;
    align-items: center;
    padding: 10px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);

    & input {
        width: calc(100% - 40px);
        height: 50px;
        margin-right: 20px;
        font-size: 24px;
        border: none;
    }

    & svg {
        width: 40px;
        height: 40px;
        cursor: pointer;
    }
`;

const Search = () => {
    const [policyURL, setPolicyURL] = useState('');

    const startAnalysis = () => {
        axios
            .post('/policy/load', { url: policyURL })
            .then(res => {
                console.log(res);
            })
            .catch(err => console.error(err));
    };

    return (
        <Container>
            <input
                type='text'
                placeholder="Analyze a website's privacy policy"
                autoFocus={true}
                onChange={(event: React.ChangeEvent<HTMLInputElement>) => setPolicyURL(event.target.value)}
                onKeyPress={(event: React.KeyboardEvent<HTMLInputElement>) => {
                    if (event.key === 'Enter') {
                        startAnalysis();
                    }
                }}
            />

            <Analyze />
        </Container>
    );
};

export default Search;
